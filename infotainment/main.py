from PySide6.QtWidgets import *
from PySide6.QtCore import *
from mainUI import Ui_MainWindow
import mysql.connector


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init()

    def init(self):
        self.db = mysql.connector.connect(host='52.79.228.222', user='soowan', password='7789', database='pjtDB', auth_plugin='mysql_native_password')
        self.cur = self.db.cursor()

        # timer setting
        self.timer = QTimer()
        self.timer.setInterval(500)  # 500ms
        self.timer.timeout.connect(self.pollingQuery)

        # browser setting
        self.ui.browser.load("http://localhost:8080/")
        self.ui.lineEdit.setText("http://localhost:8080/")
        self.ui.browser.urlChanged.connect(self.urlChangedFunction)

    def start(self):
        self.timer.start()

    def pollingQuery(self):
        self.cur.execute("select * from command order by time desc limit 15")
        self.ui.logText.clear()
        for (id, time, cmd_string, arg_string, is_finish) in self.cur:
            str = "%d | %s | %6s | %6s | %4d" % (
                id, time.strftime("%Y%m%d %H:%M:%S"), cmd_string, arg_string, is_finish)
            self.ui.logText.appendPlainText(str)

    def closeEvent(self, event):
        self.cur.close()
        self.db.close()

    def insertCommand(self, cmd_string, arg_string):
        time = QDateTime().currentDateTime().toPython()
        is_finish = 0

        query = "insert into command(time, cmd_string, arg_string, is_finish) values (%s, %s, %s, %s)"
        value = (time, cmd_string, arg_string, is_finish)

        self.cur.execute(query, value)
        self.db.commit()

    def urlBack(self):
        self.ui.browser.back()

    def urlGo(self):
        self.ui.browser.forward()

    def home(self):
        self.ui.browser.load("http://localhost:8080/")
        #self.ui.browser.reload()

    def enter(self):
        self.ui.browser.load(self.ui.lineEdit.text())
        #self.ui.browser.reload()

    def reload(self):
        self.ui.browser.reload()

    def urlChangedFunction(self):
        self.ui.lineEdit.setText(self.ui.browser.url().toString())

    def go(self):
        self.insertCommand("go", "0")

    def stop(self):
        self.insertCommand("stop", "0")

    def back(self):
        self.insertCommand("back", "0")

    def left(self):
        self.insertCommand("left", "0")

    def mid(self):
        self.insertCommand("mid", "0")

    def right(self):
        self.insertCommand("right", "0")

    def p1(self):
        self.insertCommand("place1", "0")

    def p3(self):
        self.insertCommand("place3", "0")


app = QApplication()
win = MyApp()
win.show()
app.exec()
