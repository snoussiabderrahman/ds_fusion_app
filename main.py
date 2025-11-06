import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Pour que l'icône s'affiche (sur Windows)
    try:
        from PyQt5.QtWinExtras import QtWin
        myappid = 'mycompany.myproduct.subproduct.version'
        QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        pass
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()