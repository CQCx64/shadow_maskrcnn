from windows import predict_main
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    ui = predict_main.Ui_ShadowRCNN()
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())