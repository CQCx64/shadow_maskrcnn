import datetime

import cv2

import media_choose_dialog
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

import predict_main

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = predict_main.Ui_ShadowRCNN()
    ui.setupUi(main_window)
    main_window.show()

    # main_window = QMainWindow()
    # ui = media_choose_dialog.Ui_Dialog()
    # ui.setupUi(main_window, '视频')
    # main_window.show()
    sys.exit(app.exec_())