import os
import argparse
import threading
import cv2
import numpy as np
from Raspi_MotorHAT import Raspi_MotorHAT, Raspi_DCMotor
from Raspi_PWM_Servo_Driver import PWM
import mysql.connector
from threading import Timer, Lock, Thread
from time import sleep
import signal
import sys
from sense_hat import SenseHat
from time import sleep
import datetime
import importlib.util
import queue
import pyautogui    

def closeDB(signal, frame):
    print("BYE")
    mh.getMotor(2).run(Raspi_MotorHAT.RELEASE)
    cur.close()
    print("cur")
    db.close()
    print("db")
    timer.cancel()
    print("timer")
    timer2.cancel()
    print("timer2")
    pyautogui.hotkey('q')
    print("press q key")
    control_thread.stop()
    cv2.destroyAllWindows()
    print("cv2")
    sys.exit(0)

def pushQueue():
    global cur, db, command_queue
    
    lock.acquire()
    cur.execute("select * from command order by id desc limit 1")
    for (id, time, cmd_string, arg_string, is_finish) in cur:
        if is_finish == 1 : break
        command_queue.put((id, cmd_string, arg_string))

    lock.release()
     
    global timer
    timer = Timer(0.1, pushQueue)
    timer.start()

def sensing():
    global cur, db, sense

    pressure = sense.get_pressure()
    temp = sense.get_temperature()
    humidity = sense.get_humidity()

    time = datetime.datetime.now()
    num1 = round(pressure / 10000, 3)
    num2 = round(temp / 100, 2)
    num3 = round(humidity / 100, 2)
    meta_string = '0|0|0'
    is_finish = 0

    print(num1, num2, num3)
    query = "insert into sensing(time, num1, num2, num3, meta_string, is_finish) values (%s, %s, %s, %s, %s, %s)"
    value = (time, num1, num2, num3, meta_string, is_finish)

    lock.acquire()
    cur.execute(query, value)
    db.commit()
    lock.release()

    global timer2
    timer2 = Timer(1, sensing)
    timer2.start()

def go():
    myMotor.setSpeed(100)
    myMotor.run(Raspi_MotorHAT.FORWARD)

def back():
    myMotor.setSpeed(100)
    myMotor.run(Raspi_MotorHAT.BACKWARD)

def stop():
    myMotor.run(Raspi_MotorHAT.RELEASE)

def left():
    myMotor.setSpeed(200)
    pwm.setPWM(0, 0, 280)

def mid():
    pwm.setPWM(0, 0, 375)

def right():
    myMotor.setSpeed(200)
    pwm.setPWM(0, 0, 450)


def detectObject():
    global cur, db, timer, flagDetect
    class VideoStream:
        """Camera object that controls video streaming from the Picamera"""

        def __init__(self, resolution=(640, 480), framerate=30):
            # Initialize the PiCamera and the camera image stream
            self.stream = cv2.VideoCapture(0)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])

            # Read first frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

            # Variable to control when the camera is stopped
            self.stopped = False

        def start(self):
            # Start the thread that reads frames from the video stream
            Thread(target=self.update, args=()).start()
            return self

        def update(self):
            # Keep looping indefinitely until the thread is stopped
            while True:
                # If the camera is stopped, stop the thread
                if self.stopped:
                    # Close camera resources
                    self.stream.release()
                    return

                # Otherwise, grab the next frame from the stream
                (self.grabbed, self.frame) = self.stream.read()

        def read(self):
            # Return the most recent frame
            return self.frame

        def stop(self):
            # Indicate that the camera and thread should be stopped
            self.stopped = True

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution',
                        help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del (labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
    sleep(1)

    while True:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[
            0]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        lock.acquire()
        flagDetect=0;
        lock.release()
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                print(object_name)
                
                if object_name=="person":
                  lock.acquire() 
                  flagDetect=1
                  lock.release()
                  
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text
                            
        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                    cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        
        if cv2.waitKey(1) == ord('c'):
          break
          
def place1():
    p1list = [["go",1],["stop",1],["go",0.1],["left",3],["stop",0.1],["back",0.1],["right",1.8],["stop",0.1],["mid",0.1],["back",3.5],["stop",0.1]]
    
    for i in p1list:
      time = datetime.datetime.now()
      query = "insert into command(time, cmd_string, arg_string, is_finish) values (%s, %s, %s, %s)"
      value = (time, i[0],i[1],0)
      lock.acquire()
      cur.execute(query, value)
      db.commit()
      lock.release()
      sleep(0.1)
      
def place3():
    p3list = [["go",1],["stop",1],["go",0.1],["left",4],["stop",0.1],["back",0.1],["right",0.9],["stop",0.1],["mid",0.1],["back",3.5],["stop",0.1]]
    
    for i in p3list:
      time = datetime.datetime.now()
      query = "insert into command(time, cmd_string, arg_string, is_finish) values (%s, %s, %s, %s)"
      value = (time, i[0],i[1],0)
      lock.acquire()
      cur.execute(query, value)
      db.commit()
      lock.release()
      sleep(0.1)
            
def control():
  global cur, db, command_queue, flagControl, flagBack
  while True:
    sec="0"
    if flagControl==1:
      if command_queue.qsize()==0:continue
      id, cmd,sec = command_queue.get()
      if cmd == "go" : 
        go()
        flagBack=0
      if cmd == "back" : 
        back()
        flagBack=1
      if cmd == "stop" : stop()
      if cmd == "left" : left()
      if cmd == "mid" : mid()
      if cmd == "right" : right()
      if cmd == "place1" : place1()
      if cmd == "place3" : place3()
      
      lock.acquire()
      cur.execute('update command set is_finish=1 where is_finish=0 and id={0}'.format(id))
      db.commit()
      lock.release()
      
      if sec=="0":continue
      sleep(float(sec))

#init
db = mysql.connector.connect(host='52.79.228.222', user='soowan', password='7789', database='pjtDB', auth_plugin='mysql_native_password')
cur = db.cursor()
timer = None
flagDetect = 0
flagUltra = 0
flagControl = 1
flagBack=0

mh = Raspi_MotorHAT(addr=0x6f)
myMotor = mh.getMotor(2)
pwm = PWM(0x6F)
pwm.setPWMFreq(60)

sense = SenseHat()
timer2 = None
lock = Lock()

command_queue = queue.Queue()

signal.signal(signal.SIGINT, closeDB)
pushQueue()
sensing()

dectect_thread = threading.Thread(target=detectObject,name='dectect_thread')
control_thread = threading.Thread(target=control,name='control_thread')
dectect_thread.start()
control_thread.start()

#main thread
while True:
  if flagBack==1 and flagControl == 1 and flagDetect==1:
    myMotor.run(Raspi_MotorHAT.RELEASE)
    time = datetime.datetime.now()
    is_finish = 1
    cmd_string = "stop" 
    arg_string = "0"
    query = "insert into command(time, cmd_string, arg_string, is_finish) values (%s, %s, %s, %s)"
    value = (time, cmd_string, arg_string, is_finish)
    lock.acquire()
    cur.execute(query, value)
    db.commit()
    lock.release()
    flagControl=0
    
  
  if flagBack==1 and flagDetect==0 and flagControl==0:
    flagControl=1
    time = datetime.datetime.now()
    if command_queue.qsize()==0:
      query = "insert into command(time, cmd_string, arg_string, is_finish) values (%s, %s, %s, %s)"
      value = (time, "back",1,0)
      lock.acquire()
      cur.execute(query, value)
      db.commit()
      lock.release()
      
  sleep(0.1)

    

