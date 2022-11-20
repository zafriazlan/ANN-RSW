import tensorflow as tf
import levenberg_marquardt_new as lm 
import pandas as pd 
import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from pandas import read_csv

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1023, 584)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/newPrefix/cropped-SMRI-logo-6.jpeg"))
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.tabWidget.addTab(self.tab, "")
        self.Tab2 = QtWidgets.QWidget()
        self.Tab2.setObjectName("Tab2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.Tab2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.scrollArea = QtWidgets.QScrollArea(self.Tab2)
        self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 962, 506))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setSpacing(15)
        self.formLayout.setObjectName("formLayout")
        self.label_4 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.pushButton = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.formLayout.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout)
        self.label_5 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_5.setObjectName("label_5")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_6.setObjectName("label_6")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_6)
        self.label_7 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.comboBox = QtWidgets.QComboBox(self.scrollAreaWidgetContents)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout_2.addWidget(self.comboBox)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.pushButton_2 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.formLayout.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_2)
        self.label_8 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_8.setObjectName("label_8")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.progressBar = QtWidgets.QProgressBar(self.scrollAreaWidgetContents)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.progressBar)
        self.label_9 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_9.setObjectName("label_9")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.label_14 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_14)
        self.label_10 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_10.setObjectName("label_10")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.label_15 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_15.setObjectName("label_15")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.label_15)
        self.label_11 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_11.setObjectName("label_11")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.label_16 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_16.setObjectName("label_16")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.label_16)
        self.label_13 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.radioButton_2 = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout_3.addWidget(self.radioButton_2)
        self.radioButton = QtWidgets.QRadioButton(self.scrollAreaWidgetContents)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_3.addWidget(self.radioButton)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.pushButton_4 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.formLayout.setLayout(7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_3)
        self.widget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget.setObjectName("widget")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.widget)
        self.label_17 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.lineEdit = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_4.addWidget(self.lineEdit)
        spacerItem3 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem3)
        self.formLayout.setLayout(9, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_4)
        self.label_18 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_18.setObjectName("label_18")
        self.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_5.addWidget(self.lineEdit_2)
        spacerItem4 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem4)
        self.formLayout.setLayout(10, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_5)
        self.label_19 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_19.setObjectName("label_19")
        self.formLayout.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.scrollAreaWidgetContents)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.horizontalLayout_6.addWidget(self.lineEdit_3)
        spacerItem5 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem5)
        self.formLayout.setLayout(11, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem6)
        self.pushButton_6 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_7.addWidget(self.pushButton_6)
        self.pushButton_5 = QtWidgets.QPushButton(self.scrollAreaWidgetContents)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_7.addWidget(self.pushButton_5)
        self.formLayout.setLayout(12, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_7)
        self.label_20 = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.label_20.setObjectName("label_20")
        self.formLayout.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.gridLayout_3.addLayout(self.formLayout, 0, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea, 0, 0, 1, 1)
        self.tabWidget.addTab(self.Tab2, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.scrollArea_2 = QtWidgets.QScrollArea(self.tab_2)
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 979, 506))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setSpacing(15)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_12 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_12.setObjectName("label_12")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.label_21 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_21.setObjectName("label_21")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.label_22 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_22.setObjectName("label_22")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.label_23 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_23.setObjectName("label_23")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_23)
        self.label_24 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_24.setObjectName("label_24")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_24)
        self.label_25 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_25.setObjectName("label_25")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_25)
        self.label_26 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_26.setObjectName("label_26")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_26)
        self.label_27 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_27.setObjectName("label_27")
        self.formLayout_2.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_27)
        self.label_28 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_28.setObjectName("label_28")
        self.formLayout_2.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_28)
        self.label_29 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_29.setObjectName("label_29")
        self.formLayout_2.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.label_29)
        self.label_30 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_30.setObjectName("label_30")
        self.formLayout_2.setWidget(11, QtWidgets.QFormLayout.LabelRole, self.label_30)
        self.label_31 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_31.setObjectName("label_31")
        self.formLayout_2.setWidget(13, QtWidgets.QFormLayout.LabelRole, self.label_31)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem7)
        self.pushButton_7 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalLayout_8.addWidget(self.pushButton_7)
        self.formLayout_2.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_8)
        self.label_32 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_32.setObjectName("label_32")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.label_32)
        self.label_33 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_33.setObjectName("label_33")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.label_33)
        self.label_34 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_34.setObjectName("label_34")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.label_34)
        self.label_35 = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.label_35.setObjectName("label_35")
        self.formLayout_2.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.label_35)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.comboBox_2 = QtWidgets.QComboBox(self.scrollAreaWidgetContents_2)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.horizontalLayout_9.addWidget(self.comboBox_2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem8)
        self.pushButton_8 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_9.addWidget(self.pushButton_8)
        self.formLayout_2.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_9)
        self.progressBar_2 = QtWidgets.QProgressBar(self.scrollAreaWidgetContents_2)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.progressBar_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem9)
        self.pushButton_9 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_9.setObjectName("pushButton_9")
        self.horizontalLayout_10.addWidget(self.pushButton_9)
        self.pushButton_10 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_10.setObjectName("pushButton_10")
        self.horizontalLayout_10.addWidget(self.pushButton_10)
        self.formLayout_2.setLayout(7, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.lineEdit_6 = QtWidgets.QLineEdit(self.scrollAreaWidgetContents_2)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.horizontalLayout_11.addWidget(self.lineEdit_6)
        spacerItem10 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem10)
        self.formLayout_2.setLayout(9, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.scrollAreaWidgetContents_2)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_12.addWidget(self.lineEdit_4)
        spacerItem11 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem11)
        self.formLayout_2.setLayout(10, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.lineEdit_5 = QtWidgets.QLineEdit(self.scrollAreaWidgetContents_2)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_13.addWidget(self.lineEdit_5)
        spacerItem12 = QtWidgets.QSpacerItem(800, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem12)
        self.formLayout_2.setLayout(11, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_13)
        self.widget_2 = QtWidgets.QWidget(self.scrollAreaWidgetContents_2)
        self.widget_2.setObjectName("widget_2")
        self.formLayout_2.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.widget_2)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem13)
        self.pushButton_11 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_11.setObjectName("pushButton_11")
        self.horizontalLayout_15.addWidget(self.pushButton_11)
        self.pushButton_12 = QtWidgets.QPushButton(self.scrollAreaWidgetContents_2)
        self.pushButton_12.setObjectName("pushButton_12")
        self.horizontalLayout_15.addWidget(self.pushButton_12)
        self.formLayout_2.setLayout(12, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_15)
        self.gridLayout_5.addLayout(self.formLayout_2, 0, 0, 1, 1)
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_4.addWidget(self.scrollArea_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.pushButton_5.clicked.connect(self.lineEdit.clear)
        self.pushButton_5.clicked.connect(self.lineEdit_2.clear)
        self.pushButton_5.clicked.connect(self.lineEdit_3.clear)
        self.pushButton_12.clicked.connect(self.lineEdit_6.clear)
        self.pushButton_12.clicked.connect(self.lineEdit_4.clear)
        self.pushButton_12.clicked.connect(self.lineEdit_5.clear)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.pushButton.clicked.connect(self.openFile)
        self.pushButton_2.clicked.connect(self.optim)
        self.pushButton_2.clicked.connect(self.train)    
        self.pushButton_6.clicked.connect(self.pred)  
        self.pushButton_5.clicked.connect(self.clear)
        self.pushButton_4.clicked.connect(self.plot_graph)
        self.pushButton_3.clicked.connect(self.clear_canvas)
        
        self.pushButton_7.clicked.connect(self.openFile2)
        self.pushButton_8.clicked.connect(self.optim2)
        self.pushButton_8.clicked.connect(self.train2)
        self.pushButton_11.clicked.connect(self.pred2)
        self.pushButton_12.clicked.connect(self.clear2)
        self.pushButton_9.clicked.connect(self.plot_graph2)
        self.pushButton_10.clicked.connect(self.clear_canvas2)
        
        self.lay = QHBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.lay.addWidget(self.canvas)
        self.widget.setLayout(self.lay)

        #adjust canvas size
        self.widget.setFixedWidth(610)
        self.widget.setFixedHeight(490)
 
        self.lay_2 = QHBoxLayout()
        self.figure_2 = plt.figure()
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.lay_2.addWidget(self.canvas_2)
        self.widget_2.setLayout(self.lay_2)
 
        self.widget_2.setFixedWidth(610)
        self.widget_2.setFixedHeight(490)
 
    def openFile (self):

        url = QFileDialog.getOpenFileName(self.pushButton,"Open a file","","All Files(*);;*CSV") #import directory file url
        fileUrl=url[0]
        datasets = pd.read_csv(fileUrl) #import data from csv file
        
        self.label_4.setText("CSV File Imported!")
        self.label_6.setText(fileUrl)
        
        x = datasets.iloc[:, :-1 ].values #or [:, [2,3]] 
        y = datasets.iloc[:, 3].values
        
        label_encoder_y = LabelEncoder()
        y = label_encoder_y.fit_transform(y)
        
        x_train, x_test, self.y_train, self.y_test = train_test_split(x,y, test_size =0.20, random_state = 45)
        self.sc_x = StandardScaler()
        self.x_train = self.sc_x.fit_transform(x_train)
        self.x_test = self.sc_x.transform(x_test)
        
    def openFile2 (self):
      
        url = QFileDialog.getOpenFileName(self.pushButton,"Open a file","","All Files(*);;*CSV") #import directory file url
        fileUrl=url[0]
        datasets = pd.read_csv(fileUrl) #import data from csv file
      
        self.label_12.setText("CSV File Imported!")
        self.label_32.setText(fileUrl)

        #Seperate dataset, x is input, y is output        
        #[:,0:3] means that index 0 to 2 from the imported dataset are defined as input
        #[:, 3] means that index 3 rom the imported dataset is defined as output

        x = datasets.iloc[:,0:3].values #or [:, [2,3]] 
        y = datasets.iloc[:, 3].values
      
        #label_encoder_y = LabelEncoder()
        #y = label_encoder_y.fit_transform(y)
      
        x_train, x_test, self.y_train, self.y_test = train_test_split(x,y, test_size =0.2, random_state = 45)
        
        #test_size=0.20 means that 20 percent from the dataset will used for test and the rest (80% from dataset) will be used to train
        
        self.sc_x = StandardScaler()
        self.sc_y = StandardScaler()
        
        self.x_train = self.sc_x.fit_transform(x_train)
        self.x_test = self.sc_x.transform(x_test)

        y_train = self.y_train.reshape(-1,1)
        y_test = self.y_test.reshape(-1,1)

        self.y_train = self.sc_y.fit_transform(y_train)
        self.y_test = self.sc_y.transform(y_test)

#=====================================================================================================================================================   
    def optim(self,comboBox):
        
        if self.comboBox.currentText()=="Levenberg Marquardt (LM)":
            self.val=1
            self.var=True
        elif self.comboBox.currentText()=="Gradient Descent (GD)":   
            self.val=28
            self.var=False
        else:
            self.val=1
            self.var=True
            
    def optim2(self,comboBox_2):
        
        if self.comboBox.currentText()=="Levenberg Marquardt (LM)":
            self.val=1
            self.var=True
        elif self.comboBox.currentText()=="Gradient Descent (GD)":   
            self.val=28
            self.var=False
        else:
            self.val=1
            self.var=True
#=====================================================================================================================================================   
    
    def train(self):
    
        self.classifier = Sequential()
        self.classifier.add(Dense(units=39 , input_shape = (3,), activation='relu'))
        self.classifier.add(Dense(units=39 , activation='relu'))
        self.classifier.add(Dense(units=1, activation='sigmoid'))
        self.classifier.compile(optimizer=SGD(learning_rate=0.01), loss= 'binary_crossentropy', metrics= ['binary_accuracy']) 
    
        if self.comboBox.currentText()=="Levenberg Marquardt (LM)":
        
            self.model_wrapper = lm.ModelWrapper(tf.keras.models.clone_model(self.classifier))
            self.model_wrapper.compile(
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss=lm.BinaryCrossentropy(from_logits=True),
                        metrics=['binary_accuracy'])  
        else:
            pass

        if self.comboBox.currentText()=="Levenberg Marquardt (LM)":
        
            epoc = 50
            self.progressBar.setMaximum(epoc-1)
            self.remains = []
            self.histo = None
            t1_start = time.perf_counter()
            
            for e in range(epoc):
            
                self.histo = self.model_wrapper.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1, batch_size=self.val, verbose=1, shuffle=self.var) 
                self.remains.append(self.histo.history)
                self.progressBar.setValue(e) 
                
                if self.progressBar.text() == "100%":
                    
                    scores_0 = self.model_wrapper.history.history['binary_accuracy']
                    scores_0 = np.around((max(scores_0)*100), decimals= 3)
                    self.label_12.setText(str(scores_0)+"%") #TRAIN ACCURACY
                    
                    testacc = self.model_wrapper.evaluate(self.x_test, self.y_test)
                    self.content = testacc[1]*100
                    self.label_16.setText(str(np.around(self.content, decimals = 3))+"%") #TEST ACCURACY   
            
            t1_stop = time.perf_counter()
            self.label_14.setText(str(np.around((t1_stop - t1_start), decimals=2))+' seconds') #TIME LAPSE
           
        else:
                epoc = 100
                self.progressBar.setMaximum(epoc-1)
                self.remains = []
                self.histo = None
                t1_start = time.perf_counter()
                
                for e in range(epoc):
                    
                    self.histo = self.classifier.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1, batch_size=self.val, verbose=1, shuffle=self.var)
                    self.remains.append(self.histo.history)
                    self.progressBar.setValue(e) 

                if self.progressBar.text() == "100%":
                    
                    scores_0 = self.classifier.history.history['binary_accuracy']
                    scores_0 = np.around((max(scores_0)*100), decimals= 3)
                    self.label_15.setText(str(scores_0)+"%") #TRAIN ACCURACY 
                    
                    scores = self.classifier.evaluate(self.x_test, self.y_test)
                    self.content = scores[1]*100
                    self.label_16.setText(str(np.around(self.content, decimals = 3))+"%") #TEST ACCURACY
                    
                t1_stop = time.perf_counter()
                self.label_14.setText(str(np.around((t1_stop - t1_start), decimals=2))+' seconds') #TIME LAPSE

    def train2(self):
    
        self.model = Sequential()
        self.model.add(Dense(units=6 , input_shape = (3,), activation='relu'))
        self.model.add(Dense(units=12 , activation='relu'))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(optimizer=SGD(learning_rate=0.01), loss= 'MSE')
        
        if self.comboBox_2.currentText()=="Levenberg Marquardt (LM)":
        
            self.model_wrapper = lm.ModelWrapper(tf.keras.models.clone_model(self.model))
            self.model_wrapper.compile(
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                        loss=lm.MeanSquaredError(from_logits=True),
                        metrics=['']) 
        else:
            pass
        
        if self.comboBox_2.currentText()=="Levenberg Marquardt (LM)":
        
            epoc = 50
            self.progressBar_2.setMaximum(epoc-1)
            self.remains2 = []
            self.histo2 = None
            t1_start2 = time.perf_counter()
            
            for e in range(epoc):
            
                self.histo2 = self.model_wrapper.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1, batch_size=self.val, shuffle=True) 
                self.remains2.append(self.histo2.history)
                self.progressBar_2.setValue(e) 
                
                if self.progressBar_2.text() == "100%":
                    
                    scores_0 = self.model_wrapper.history.history['loss']
                    scores_0 = np.around((max(scores_0)),decimals=4)
                    self.label_34.setText(str((scores_0))) #loss 
                    
                    scores_1 = self.model_wrapper.evaluate(self.x_test, self.y_test)
                    scores_1 = np.around((max(scores_1)),decimals=4)
                    self.label_35.setText(str((scores_1)))  #validation loss
            
            t1_stop2 = time.perf_counter()
            self.label_33.setText(str(np.around((t1_stop2 - t1_start2), decimals=2))+' seconds') #TIME LAPSE

        else:        

             epoc = 100
             self.progressBar_2.setMaximum(epoc-1)
             self.remains2 = []
             self.histo2 = None
             t1_start2 = time.perf_counter()
                
             for e in range(epoc):
                    
                 self.histo2 = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=1, batch_size=self.val, shuffle=True)
                 self.remains2.append(self.histo2.history)
                 self.progressBar_2.setValue(e) 

             if self.progressBar_2.text() == "100%":
                    
                 scores_0 = self.model.history.history['loss']
                 scores_0 = np.around((max(scores_0)),decimals=4)
                 self.label_34.setText(str((scores_0))) 
                           
                 scores_1 = self.model.history.history['val_loss']
                 scores_1 = np.around((max(scores_1)),decimals=4)
                 self.label_35.setText(str((scores_1)))
                
             t1_stop2 = time.perf_counter()
             self.label_33.setText(str(np.around((t1_stop2 - t1_start2), decimals=2))+' seconds') #TIME LAPSE
                
#=======================================================================================================================================================================                
             
    def plot_graph(self):
        
        df = pd.DataFrame(self.remains)
        
        if self.radioButton_2.isChecked():
            
            self.figure.clear()
            
            df['loss'] = df['loss'].str.get(0)
            df['val_loss'] = df['val_loss'].str.get(0)
            
            plt.plot(df['loss'])
            plt.plot(df['val_loss'])
            
            plt.title('Learning curve', loc='right', weight='bold')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        elif self.radioButton.isChecked():
            self.figure.clear()
            
            df['binary_accuracy'] = df['binary_accuracy'].str.get(0)
            df['val_binary_accuracy'] = df['val_binary_accuracy'].str.get(0)
            
            plt.plot(df['binary_accuracy'])
            plt.plot(df['val_binary_accuracy'])
            
            plt.title('Model Accuracy', loc='right', weight='bold')
            
            self.figure.tight_layout()
            self.canvas.draw()
        else:
            pass
   
    def plot_graph2(self):
        df = pd.DataFrame(self.remains2)
        
        self.figure_2.clear()
            
        df['loss'] = df['loss'].str.get(0)
        df['val_loss'] = df['val_loss'].str.get(0)
            
        plt.plot(df['loss'])
        plt.plot(df['val_loss'])
            
        plt.title('Loss curve', loc='right', weight='bold')
        
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')

        
        self.figure_2.tight_layout()
        self.canvas_2.draw()
        
#=======================================================================================================================================        
    #FUNCTION FOR BUTTON SUBMIT TO START PREDICT    
    def pred(self):

        self.forc=float(self.lineEdit.text())
        self.tim=float(self.lineEdit_2.text())
        self.curr=float(self.lineEdit_3.text())
        
        pred_val = np.array([[self.tim,self.curr,self.forc]]) 
        pred_val = self.sc_x.transform(pred_val)

        if self.comboBox.currentText()=="Levenberg Marquardt (LM)":
            y_pred = self.model_wrapper.predict(pred_val)
        else:
            y_pred = self.classifier.predict(pred_val)  

        y_pred = y_pred.round()   
        
        if y_pred == 0: # ZERO WILL REPRESENT FOR BAD
            self.label_20.setText("Result : BAD")
        if y_pred == 1: # ONE WILL REPRESENT FOR GOOD
            self.label_20.setText("Result : GOOD") 
 
    def pred2(self):

        self.forc=float(self.lineEdit_6.text())
        self.tim=float(self.lineEdit_4.text())
        self.curr=float(self.lineEdit_5.text())
        
        pred_val = np.array([[self.tim,self.curr,self.forc]]) 
        pred_val = self.sc_x.transform(pred_val)
        pred_val = pred_val.reshape(1,-1)
        
        y_pred = self.model.predict(pred_val)
        
        y_pred = self.sc_y.inverse_transform(y_pred)
        y_pred = y_pred[0]
        
        y_pred=str(np.around((y_pred[0]), decimals=4))+' kN'
        
        self.label_31.setText("Tensile load: " + (y_pred))
#=====================================================================================================================================            
            
    def clear(self):
          
          self.label_5.setText("Result: ")
 
    def clear2(self):
          
          self.label_31.setText("Tensile load:")    
#=====================================================================================================================================          
    #rEMOVE PREVIOUS GRAPH
    def clear_canvas(self):
    
        self.figure.clear()
        self.canvas.draw()

    def clear_canvas2(self):
    
        self.figure_2.clear()
        self.canvas_2.draw()        
#=======================================================================================    


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ANN For Resistance Spot Welding"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">APPLICATION OF ARTIFICIAL NEURAL NETWORK(ANN) USING PYTHON LANGUAGE</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">MANUFACTURING RESEARCH FIELD: RESISTANCE SPOT WELDING</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">(QUALITATIVE &amp; QUANTITATIVE METHOD) </span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">DEVELOPED BY:</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">PROF. IR. TS. DR-ING. YUPITER HP MANURUNG</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">TS. DR. SUHAILA ABD HALIM</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">ALIF HAIKAL</span></p><p align=\"center\"><span style=\" font-family:\'Times New Roman,serif\'; font-size:18pt; font-weight:600;\">MOHAMAD ZAFRI</span></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Introduction"))
        self.label_4.setText(_translate("MainWindow", "Import the Binary CSV File"))
        self.pushButton.setText(_translate("MainWindow", "Open file"))
        self.label_5.setText(_translate("MainWindow", "File URL:"))
        self.label_6.setText(_translate("MainWindow", "-"))
        self.label_7.setText(_translate("MainWindow", "Optimizer"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Gradient Descent (GD)"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Stochastic Gradient Descent (SGD)"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Lavenberg Marquardt (LM)"))
        self.pushButton_2.setText(_translate("MainWindow", "Train"))
        self.label_8.setText(_translate("MainWindow", "Training progress"))
        self.label_9.setText(_translate("MainWindow", "Time lapse:"))
        self.label_14.setText(_translate("MainWindow", "-"))
        self.label_10.setText(_translate("MainWindow", "Train accuracy:"))
        self.label_15.setText(_translate("MainWindow", "-"))
        self.label_11.setText(_translate("MainWindow", "Test accuracy:"))
        self.label_16.setText(_translate("MainWindow", "-"))
        self.label_13.setText(_translate("MainWindow", "Model graph"))
        self.radioButton_2.setText(_translate("MainWindow", "Loss"))
        self.radioButton.setText(_translate("MainWindow", "Accuracy"))
        self.pushButton_4.setText(_translate("MainWindow", "Plot"))
        self.pushButton_3.setText(_translate("MainWindow", "Clear"))
        self.label_17.setText(_translate("MainWindow", "Force (kgf)"))
        self.label_18.setText(_translate("MainWindow", "Weld time (Cycle time)"))
        self.label_19.setText(_translate("MainWindow", "Weld current (kA)"))
        self.pushButton_6.setText(_translate("MainWindow", "Predict"))
        self.pushButton_5.setText(_translate("MainWindow", "Clear"))
        self.label_20.setText(_translate("MainWindow", "Result:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Tab2), _translate("MainWindow", "Prediction (Qualitative method)"))
        self.label_12.setText(_translate("MainWindow", "Import the Regression CSV File"))
        self.label_21.setText(_translate("MainWindow", "File URL:"))
        self.label_22.setText(_translate("MainWindow", "Optimizer"))
        self.label_23.setText(_translate("MainWindow", "Training progress"))
        self.label_24.setText(_translate("MainWindow", "Time lapse:"))
        self.label_25.setText(_translate("MainWindow", "Loss:"))
        self.label_26.setText(_translate("MainWindow", "Validation loss:"))
        self.label_27.setText(_translate("MainWindow", "Model graph"))
        self.label_28.setText(_translate("MainWindow", "Force (kgf)"))
        self.label_29.setText(_translate("MainWindow", "Weld time (Cycle time)"))
        self.label_30.setText(_translate("MainWindow", "Weld current (kA)"))
        self.label_31.setText(_translate("MainWindow", "Tensile load:"))
        self.pushButton_7.setText(_translate("MainWindow", "Open file"))
        self.label_32.setText(_translate("MainWindow", "-"))
        self.label_33.setText(_translate("MainWindow", "-"))
        self.label_34.setText(_translate("MainWindow", "-"))
        self.label_35.setText(_translate("MainWindow", "-"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Gradient Descent (GD)"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Stochastic Gradient Descent (SGD)"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Lavenberg Marquardt (LM)"))
        self.pushButton_8.setText(_translate("MainWindow", "Train"))
        self.pushButton_9.setText(_translate("MainWindow", "Plot"))
        self.pushButton_10.setText(_translate("MainWindow", "Clear"))
        self.pushButton_11.setText(_translate("MainWindow", "Predict"))
        self.pushButton_12.setText(_translate("MainWindow", "Clear"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Prediction (Quantitative method)"))

import rsc_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

