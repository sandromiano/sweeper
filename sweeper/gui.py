from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg
import numpy as np

class sweeperGUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('sweeperGUI')
