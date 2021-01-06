# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Annotation_Gui.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(937, 745)
        self.actionrecordfile = QAction(MainWindow)
        self.actionrecordfile.setObjectName(u"actionrecordfile")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(180, 20, 20, 691))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.videolabel = QLabel(self.centralwidget)
        self.videolabel.setObjectName(u"videolabel")
        self.videolabel.setGeometry(QRect(210, 180, 360, 400))
        self.videolabel.setFrameShadow(QFrame.Sunken)
        self.videolabel.setLineWidth(5)
        self.videolabel.setMargin(0)
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(580, 10, 20, 691))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.layoutWidget1 = QWidget(self.centralwidget)
        self.layoutWidget1.setObjectName(u"layoutWidget1")
        self.layoutWidget1.setGeometry(QRect(10, 10, 171, 681))
        self.verticalLayout = QVBoxLayout(self.layoutWidget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.layoutWidget1)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.wordexplore = QLineEdit(self.layoutWidget1)
        self.wordexplore.setObjectName(u"wordexplore")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.wordexplore)


        self.verticalLayout.addLayout(self.formLayout)

        self.listWidget = QListWidget(self.layoutWidget1)
        self.listWidget.setObjectName(u"listWidget")

        self.verticalLayout.addWidget(self.listWidget)

        self.layoutWidget2 = QWidget(self.centralwidget)
        self.layoutWidget2.setObjectName(u"layoutWidget2")
        self.layoutWidget2.setGeometry(QRect(230, 590, 331, 103))
        self.gridLayout = QGridLayout(self.layoutWidget2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.layoutWidget2)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)

        self.endindex_spinBox = QSpinBox(self.layoutWidget2)
        self.endindex_spinBox.setObjectName(u"endindex_spinBox")

        self.gridLayout.addWidget(self.endindex_spinBox, 1, 2, 1, 1)

        self.beginindex_Slider = QSlider(self.layoutWidget2)
        self.beginindex_Slider.setObjectName(u"beginindex_Slider")
        self.beginindex_Slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.beginindex_Slider, 0, 1, 1, 1)

        self.beginindex_spinBox = QSpinBox(self.layoutWidget2)
        self.beginindex_spinBox.setObjectName(u"beginindex_spinBox")

        self.gridLayout.addWidget(self.beginindex_spinBox, 0, 2, 1, 1)

        self.endindex_Slider = QSlider(self.layoutWidget2)
        self.endindex_Slider.setObjectName(u"endindex_Slider")
        self.endindex_Slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.endindex_Slider, 1, 1, 1, 1)

        self.label_4 = QLabel(self.layoutWidget2)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)

        self.layoutWidget3 = QWidget(self.centralwidget)
        self.layoutWidget3.setObjectName(u"layoutWidget3")
        self.layoutWidget3.setGeometry(QRect(610, 600, 265, 89))
        self.gridLayout_2 = QGridLayout(self.layoutWidget3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.btn_addneg = QPushButton(self.layoutWidget3)
        self.btn_addneg.setObjectName(u"btn_addneg")

        self.gridLayout_2.addWidget(self.btn_addneg, 0, 1, 1, 1)

        self.btn_addpos = QPushButton(self.layoutWidget3)
        self.btn_addpos.setObjectName(u"btn_addpos")

        self.gridLayout_2.addWidget(self.btn_addpos, 0, 0, 1, 1)

        self.btn_slowdown = QPushButton(self.layoutWidget3)
        self.btn_slowdown.setObjectName(u"btn_slowdown")

        self.gridLayout_2.addWidget(self.btn_slowdown, 2, 1, 1, 1)

        self.btn_nextword = QPushButton(self.layoutWidget3)
        self.btn_nextword.setObjectName(u"btn_nextword")

        self.gridLayout_2.addWidget(self.btn_nextword, 1, 0, 1, 1)

        self.btn_speedup = QPushButton(self.layoutWidget3)
        self.btn_speedup.setObjectName(u"btn_speedup")

        self.gridLayout_2.addWidget(self.btn_speedup, 2, 0, 1, 1)

        self.btn_nextsample = QPushButton(self.layoutWidget3)
        self.btn_nextsample.setObjectName(u"btn_nextsample")

        self.gridLayout_2.addWidget(self.btn_nextsample, 1, 1, 1, 1)

        self.shapelet_video = QLabel(self.centralwidget)
        self.shapelet_video.setObjectName(u"shapelet_video")
        self.shapelet_video.setGeometry(QRect(610, 180, 300, 360))
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.shapelet_video.sizePolicy().hasHeightForWidth())
        self.shapelet_video.setSizePolicy(sizePolicy)
        self.line_4 = QFrame(self.centralwidget)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setGeometry(QRect(200, 110, 361, 16))
        self.line_4.setLineWidth(2)
        self.line_4.setFrameShape(QFrame.HLine)
        self.line_4.setFrameShadow(QFrame.Sunken)
        self.layoutWidget4 = QWidget(self.centralwidget)
        self.layoutWidget4.setObjectName(u"layoutWidget4")
        self.layoutWidget4.setGeometry(QRect(200, 10, 361, 89))
        self.gridLayout_3 = QGridLayout(self.layoutWidget4)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.videokey_text = QLineEdit(self.layoutWidget4)
        self.videokey_text.setObjectName(u"videokey_text")

        self.gridLayout_3.addWidget(self.videokey_text, 0, 3, 1, 1)

        self.label_10 = QLabel(self.layoutWidget4)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_3.addWidget(self.label_10, 2, 2, 1, 1)

        self.label_7 = QLabel(self.layoutWidget4)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_3.addWidget(self.label_7, 1, 2, 1, 1)

        self.annotated_text = QLineEdit(self.layoutWidget4)
        self.annotated_text.setObjectName(u"annotated_text")
        self.annotated_text.setEnabled(True)

        self.gridLayout_3.addWidget(self.annotated_text, 2, 3, 1, 1)

        self.offset_text = QLineEdit(self.layoutWidget4)
        self.offset_text.setObjectName(u"offset_text")

        self.gridLayout_3.addWidget(self.offset_text, 1, 1, 1, 1)

        self.label_11 = QLabel(self.layoutWidget4)
        self.label_11.setObjectName(u"label_11")

        self.gridLayout_3.addWidget(self.label_11, 2, 0, 1, 1)

        self.label_8 = QLabel(self.layoutWidget4)
        self.label_8.setObjectName(u"label_8")

        self.gridLayout_3.addWidget(self.label_8, 1, 0, 1, 1)

        self.idx_text = QLineEdit(self.layoutWidget4)
        self.idx_text.setObjectName(u"idx_text")

        self.gridLayout_3.addWidget(self.idx_text, 2, 1, 1, 1)

        self.label_6 = QLabel(self.layoutWidget4)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_3.addWidget(self.label_6, 0, 2, 1, 1)

        self.length_text = QLineEdit(self.layoutWidget4)
        self.length_text.setObjectName(u"length_text")

        self.gridLayout_3.addWidget(self.length_text, 1, 3, 1, 1)

        self.word_text = QLineEdit(self.layoutWidget4)
        self.word_text.setObjectName(u"word_text")

        self.gridLayout_3.addWidget(self.word_text, 0, 1, 1, 1)

        self.label_5 = QLabel(self.layoutWidget4)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_3.addWidget(self.label_5, 0, 0, 1, 1)

        self.layoutWidget5 = QWidget(self.centralwidget)
        self.layoutWidget5.setObjectName(u"layoutWidget5")
        self.layoutWidget5.setGeometry(QRect(601, 10, 141, 89))
        self.formLayout_2 = QFormLayout(self.layoutWidget5)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.beginidx_label = QLabel(self.layoutWidget5)
        self.beginidx_label.setObjectName(u"beginidx_label")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.beginidx_label)

        self.beginidx_text = QLineEdit(self.layoutWidget5)
        self.beginidx_text.setObjectName(u"beginidx_text")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.beginidx_text)

        self.endidx_label = QLabel(self.layoutWidget5)
        self.endidx_label.setObjectName(u"endidx_label")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.endidx_label)

        self.endidx_text = QLineEdit(self.layoutWidget5)
        self.endidx_text.setObjectName(u"endidx_text")

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.endidx_text)

        self.postive_label = QLabel(self.layoutWidget5)
        self.postive_label.setObjectName(u"postive_label")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.postive_label)

        self.positive_text = QLineEdit(self.layoutWidget5)
        self.positive_text.setObjectName(u"positive_text")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.positive_text)

        self.layoutWidget6 = QWidget(self.centralwidget)
        self.layoutWidget6.setObjectName(u"layoutWidget6")
        self.layoutWidget6.setGeometry(QRect(192, 140, 371, 27))
        self.horizontalLayout = QHBoxLayout(self.layoutWidget6)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.layoutWidget6)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.currentframe = QLineEdit(self.layoutWidget6)
        self.currentframe.setObjectName(u"currentframe")

        self.horizontalLayout.addWidget(self.currentframe)

        self.label_12 = QLabel(self.layoutWidget6)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout.addWidget(self.label_12)

        self.speed_text = QLineEdit(self.layoutWidget6)
        self.speed_text.setObjectName(u"speed_text")

        self.horizontalLayout.addWidget(self.speed_text)

        self.layoutWidget7 = QWidget(self.centralwidget)
        self.layoutWidget7.setObjectName(u"layoutWidget7")
        self.layoutWidget7.setGeometry(QRect(600, 140, 301, 27))
        self.horizontalLayout_2 = QHBoxLayout(self.layoutWidget7)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.label_9 = QLabel(self.layoutWidget7)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_2.addWidget(self.label_9)

        self.location_text = QLineEdit(self.layoutWidget7)
        self.location_text.setObjectName(u"location_text")

        self.horizontalLayout_2.addWidget(self.location_text)

        self.label_13 = QLabel(self.layoutWidget7)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_2.addWidget(self.label_13)

        self.m_len_combo = QComboBox(self.layoutWidget7)
        self.m_len_combo.setObjectName(u"m_len_combo")

        self.horizontalLayout_2.addWidget(self.m_len_combo)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 937, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuLoad = QMenu(self.menuFile)
        self.menuLoad.setObjectName(u"menuLoad")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.menuLoad.menuAction())
        self.menuLoad.addAction(self.actionrecordfile)

        self.retranslateUi(MainWindow)
        self.beginindex_Slider.valueChanged.connect(self.beginindex_spinBox.setValue)
        self.beginindex_spinBox.valueChanged.connect(self.beginindex_Slider.setValue)
        self.endindex_Slider.valueChanged.connect(self.endindex_spinBox.setValue)
        self.endindex_spinBox.valueChanged.connect(self.endindex_Slider.setValue)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Annotation & display", None))
        self.actionrecordfile.setText(QCoreApplication.translate("MainWindow", u"recordfile", None))
        self.videolabel.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"explore", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"beginindex", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"endindex:", None))
        self.btn_addneg.setText(QCoreApplication.translate("MainWindow", u"add neg", None))
        self.btn_addpos.setText(QCoreApplication.translate("MainWindow", u"add pos", None))
        self.btn_slowdown.setText(QCoreApplication.translate("MainWindow", u"SlowDown", None))
        self.btn_nextword.setText(QCoreApplication.translate("MainWindow", u"next word", None))
        self.btn_speedup.setText(QCoreApplication.translate("MainWindow", u"SpeedUp", None))
        self.btn_nextsample.setText(QCoreApplication.translate("MainWindow", u"next sample", None))
        self.shapelet_video.setText(QCoreApplication.translate("MainWindow", u"shapelet_video", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"marked", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"length", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"idx/num", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"offset", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"videokey", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"word", None))
        self.beginidx_label.setText(QCoreApplication.translate("MainWindow", u"beginidx", None))
        self.endidx_label.setText(QCoreApplication.translate("MainWindow", u"endidx", None))
        self.postive_label.setText(QCoreApplication.translate("MainWindow", u"Positive", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"current ", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"speed", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"loc", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"m_len", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuLoad.setTitle(QCoreApplication.translate("MainWindow", u"Load", None))
    # retranslateUi

