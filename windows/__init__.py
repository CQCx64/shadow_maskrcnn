import datetime

import cv2
from PyQt5 import QtCore

import login
import media_choose_dialog
import MySQLdb
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import predict_main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = predict_main.Ui_ShadowRCNN()
    # ui.setupUi(main_window)
    # main_window.show()

    ui.login(main_window)
    main_window.show()

    sys.exit(app.exec_())