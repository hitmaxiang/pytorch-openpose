# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'annotation_gui.ui'
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
        MainWindow.resize(920, 649)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(210, 30, 20, 561))
        self.line.setFrameShape(QFrame.VLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.videolabel = QLabel(self.centralwidget)
        self.videolabel.setObjectName(u"videolabel")
        self.videolabel.setGeometry(QRect(230, 80, 360, 360))
        self.videolabel.setFrameShadow(QFrame.Sunken)
        self.videolabel.setLineWidth(5)
        self.videolabel.setMargin(0)
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(680, 30, 217, 275))
        self.formLayout_3 = QFormLayout(self.widget)
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.widget)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_5)

        self.word_text = QLineEdit(self.widget)
        self.word_text.setObjectName(u"word_text")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.word_text)

        self.label_6 = QLabel(self.widget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_6)

        self.videokey_text = QLineEdit(self.widget)
        self.videokey_text.setObjectName(u"videokey_text")

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.videokey_text)

        self.label_8 = QLabel(self.widget)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_8)

        self.offset_text = QLineEdit(self.widget)
        self.offset_text.setObjectName(u"offset_text")

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.offset_text)

        self.label_7 = QLabel(self.widget)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label_7)

        self.length_text = QLineEdit(self.widget)
        self.length_text.setObjectName(u"length_text")

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.length_text)

        self.label_11 = QLabel(self.widget)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.label_11)

        self.idx_text = QLineEdit(self.widget)
        self.idx_text.setObjectName(u"idx_text")

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.idx_text)

        self.label_10 = QLabel(self.widget)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_3.setWidget(5, QFormLayout.LabelRole, self.label_10)

        self.annotated_text = QLineEdit(self.widget)
        self.annotated_text.setObjectName(u"annotated_text")
        self.annotated_text.setEnabled(True)

        self.formLayout_3.setWidget(5, QFormLayout.FieldRole, self.annotated_text)

        self.postive_label = QLabel(self.widget)
        self.postive_label.setObjectName(u"postive_label")

        self.formLayout_3.setWidget(6, QFormLayout.LabelRole, self.postive_label)

        self.positive_text = QLineEdit(self.widget)
        self.positive_text.setObjectName(u"positive_text")

        self.formLayout_3.setWidget(6, QFormLayout.FieldRole, self.positive_text)

        self.beginidx_label = QLabel(self.widget)
        self.beginidx_label.setObjectName(u"beginidx_label")

        self.formLayout_3.setWidget(7, QFormLayout.LabelRole, self.beginidx_label)

        self.beginidx_text = QLineEdit(self.widget)
        self.beginidx_text.setObjectName(u"beginidx_text")

        self.formLayout_3.setWidget(7, QFormLayout.FieldRole, self.beginidx_text)

        self.endidx_label = QLabel(self.widget)
        self.endidx_label.setObjectName(u"endidx_label")

        self.formLayout_3.setWidget(8, QFormLayout.LabelRole, self.endidx_label)

        self.endidx_text = QLineEdit(self.widget)
        self.endidx_text.setObjectName(u"endidx_text")

        self.formLayout_3.setWidget(8, QFormLayout.FieldRole, self.endidx_text)

        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(650, 30, 20, 551))
        self.line_2.setFrameShape(QFrame.VLine)
        self.line_2.setFrameShadow(QFrame.Sunken)
        self.widget1 = QWidget(self.centralwidget)
        self.widget1.setObjectName(u"widget1")
        self.widget1.setGeometry(QRect(0, 30, 201, 541))
        self.verticalLayout = QVBoxLayout(self.widget1)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.widget1)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.wordexplore = QLineEdit(self.widget1)
        self.wordexplore.setObjectName(u"wordexplore")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.wordexplore)


        self.verticalLayout.addLayout(self.formLayout)

        self.listWidget = QListWidget(self.widget1)
        self.listWidget.setObjectName(u"listWidget")

        self.verticalLayout.addWidget(self.listWidget)

        self.widget2 = QWidget(self.centralwidget)
        self.widget2.setObjectName(u"widget2")
        self.widget2.setGeometry(QRect(230, 490, 331, 103))
        self.gridLayout = QGridLayout(self.widget2)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.label_3 = QLabel(self.widget2)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)

        self.endindex_spinBox = QSpinBox(self.widget2)
        self.endindex_spinBox.setObjectName(u"endindex_spinBox")

        self.gridLayout.addWidget(self.endindex_spinBox, 1, 2, 1, 1)

        self.beginindex_Slider = QSlider(self.widget2)
        self.beginindex_Slider.setObjectName(u"beginindex_Slider")
        self.beginindex_Slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.beginindex_Slider, 0, 1, 1, 1)

        self.beginindex_spinBox = QSpinBox(self.widget2)
        self.beginindex_spinBox.setObjectName(u"beginindex_spinBox")

        self.gridLayout.addWidget(self.beginindex_spinBox, 0, 2, 1, 1)

        self.endindex_Slider = QSlider(self.widget2)
        self.endindex_Slider.setObjectName(u"endindex_Slider")
        self.endindex_Slider.setOrientation(Qt.Horizontal)

        self.gridLayout.addWidget(self.endindex_Slider, 1, 1, 1, 1)

        self.label_4 = QLabel(self.widget2)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)

        self.widget3 = QWidget(self.centralwidget)
        self.widget3.setObjectName(u"widget3")
        self.widget3.setGeometry(QRect(690, 490, 211, 89))
        self.gridLayout_2 = QGridLayout(self.widget3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.btn_addneg = QPushButton(self.widget3)
        self.btn_addneg.setObjectName(u"btn_addneg")

        self.gridLayout_2.addWidget(self.btn_addneg, 0, 1, 1, 1)

        self.btn_nextsample = QPushButton(self.widget3)
        self.btn_nextsample.setObjectName(u"btn_nextsample")

        self.gridLayout_2.addWidget(self.btn_nextsample, 1, 1, 1, 1)

        self.btn_addpos = QPushButton(self.widget3)
        self.btn_addpos.setObjectName(u"btn_addpos")

        self.gridLayout_2.addWidget(self.btn_addpos, 0, 0, 1, 1)

        self.btn_nextword = QPushButton(self.widget3)
        self.btn_nextword.setObjectName(u"btn_nextword")

        self.gridLayout_2.addWidget(self.btn_nextword, 1, 0, 1, 1)

        self.btn_speedup = QPushButton(self.widget3)
        self.btn_speedup.setObjectName(u"btn_speedup")

        self.gridLayout_2.addWidget(self.btn_speedup, 2, 0, 1, 1)

        self.btn_slowdown = QPushButton(self.widget3)
        self.btn_slowdown.setObjectName(u"btn_slowdown")

        self.gridLayout_2.addWidget(self.btn_slowdown, 2, 1, 1, 1)

        self.widget4 = QWidget(self.centralwidget)
        self.widget4.setObjectName(u"widget4")
        self.widget4.setGeometry(QRect(241, 30, 371, 27))
        self.horizontalLayout = QHBoxLayout(self.widget4)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.widget4)
        self.label_2.setObjectName(u"label_2")

        self.horizontalLayout.addWidget(self.label_2)

        self.currentframe = QLineEdit(self.widget4)
        self.currentframe.setObjectName(u"currentframe")

        self.horizontalLayout.addWidget(self.currentframe)

        self.horizontalSpacer = QSpacerItem(48, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)

        self.label_12 = QLabel(self.widget4)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout.addWidget(self.label_12)

        self.speed_text = QLineEdit(self.widget4)
        self.speed_text.setObjectName(u"speed_text")

        self.horizontalLayout.addWidget(self.speed_text)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 920, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.beginindex_Slider.valueChanged.connect(self.beginindex_spinBox.setValue)
        self.beginindex_spinBox.valueChanged.connect(self.beginindex_Slider.setValue)
        self.endindex_Slider.valueChanged.connect(self.endindex_spinBox.setValue)
        self.endindex_spinBox.valueChanged.connect(self.endindex_Slider.setValue)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Annotation & display", None))
        self.videolabel.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"word", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"videokey", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"offset", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"length", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"idx/num", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"marked", None))
        self.postive_label.setText(QCoreApplication.translate("MainWindow", u"Positive", None))
        self.beginidx_label.setText(QCoreApplication.translate("MainWindow", u"beginidx", None))
        self.endidx_label.setText(QCoreApplication.translate("MainWindow", u"endidx", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Word:", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"beginindex", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"endindex:", None))
        self.btn_addneg.setText(QCoreApplication.translate("MainWindow", u"add neg", None))
        self.btn_nextsample.setText(QCoreApplication.translate("MainWindow", u"next sample", None))
        self.btn_addpos.setText(QCoreApplication.translate("MainWindow", u"add pos", None))
        self.btn_nextword.setText(QCoreApplication.translate("MainWindow", u"next word", None))
        self.btn_speedup.setText(QCoreApplication.translate("MainWindow", u"SpeedUp", None))
        self.btn_slowdown.setText(QCoreApplication.translate("MainWindow", u"SlowDown", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"current frame:", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"speed", None))
    # retranslateUi

