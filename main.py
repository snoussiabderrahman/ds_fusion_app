import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from PyQt5.QtGui import QIcon
import sys
import os

def resource_path(relative_path: str) -> str:
    """Retourne le chemin absolu vers une ressource, dev ou bundle PyInstaller."""
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS  # chemin temporaire créé par PyInstaller
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(resource_path(os.path.join("resources", "icon.png"))))
    
    # Pour que l'icône s'affiche (sur Windows)
    try:
        from PyQt5.QtWinExtras import QtWin
        myappid = 'mycompany.myproduct.subproduct.version'
        QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
    except ImportError:
        print("QtWinExtras module not found, skipping Windows icon fix.")
        pass
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()