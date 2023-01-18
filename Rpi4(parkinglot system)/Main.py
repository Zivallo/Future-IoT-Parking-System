import paho.mqtt.client as mqtt
import mysql.connector
from threading import Timer
from time import sleep
import signal
import sys
import datetime


def closeDB(signal, frame):
    print("BYE")
    cur.close()
    db.close()
    timer.cancel()
    sys.exit(0)

def polling():
    global cur, db,p1,p3
    
    time = datetime.datetime.now()

    query = "insert into parkinglot(time, place1, place2, place3, place4) values (%s, %s, %s, %s, %s)"
    value = (time, p1, 1, p3, 1)

    cur.execute(query, value)
    db.commit()

     
    global timer
    timer = Timer(1, polling)
    timer.start()



p1 = 0
p3 = 0

db = mysql.connector.connect(host='52.79.228.222', user='soowan', password='7789', database='pjtDB', auth_plugin='mysql_native_password')
cur = db.cursor()


timer = None
signal.signal(signal.SIGINT, closeDB)
polling()

# subscriber callback
def on_message(client, userdata, message):
    #print("message received ", str(message.payload.decode("utf-8")))
    
    print("message topic=", message.topic)
   # print("message qos=", message.qos)
   # print("message retain flag=", message.retain)
    
    global p1, p3
    if(message.topic == "parkinglot/place1"):
        p1 = int(str(message.payload.decode("utf-8")))
    
    else:
        p3 = int(str(message.payload.decode("utf-8")))
    
    print("p1 = ", p1)
    print("p3 = ", p3)

broker_address = "192.168.110.201"
client1 = mqtt.Client("client1") 
client1.connect(broker_address) 
client1.subscribe("parkinglot/#")


client1.on_message = on_message
client1.loop_forever()