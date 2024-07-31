import sys
from PyQt5.QtWidgets import QApplication
from user_interface import App

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = App()
    main_window.run()
    sys.exit(app.exec_())
