# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'DisplayShapeletRecord.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(885, 235)
        self.nextinstance = QPushButton(Form)
        self.nextinstance.setObjectName(u"nextinstance")
        self.nextinstance.setGeometry(QRect(710, 40, 101, 25))
        self.checkInstance = QPushButton(Form)
        self.checkInstance.setObjectName(u"checkInstance")
        self.checkInstance.setGeometry(QRect(710, 80, 101, 25))
        self.checkAllInstances = QPushButton(Form)
        self.checkAllInstances.setObjectName(u"checkAllInstances")
        self.checkAllInstances.setGeometry(QRect(710, 120, 101, 25))
        self.widget = QWidget(Form)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(40, 20, 649, 197))
        self.horizontalLayout_4 = QHBoxLayout(self.widget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.groupBox = QGroupBox(self.widget)
        self.groupBox.setObjectName(u"groupBox")
        self.horizontalLayout = QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label = QLabel(self.groupBox)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label)

        self.wordcomboBox = QComboBox(self.groupBox)
        self.wordcomboBox.setObjectName(u"wordcomboBox")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.wordcomboBox)

        self.label_2 = QLabel(self.groupBox)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_2)

        self.levelcomboBox = QComboBox(self.groupBox)
        self.levelcomboBox.setObjectName(u"levelcomboBox")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.levelcomboBox)


        self.horizontalLayout.addLayout(self.formLayout)


        self.horizontalLayout_4.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.widget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.horizontalLayout_2 = QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_3 = QLabel(self.groupBox_2)
        self.label_3.setObjectName(u"label_3")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_3)

        self.videokeylineedit = QLineEdit(self.groupBox_2)
        self.videokeylineedit.setObjectName(u"videokeylineedit")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.videokeylineedit)

        self.label_4 = QLabel(self.groupBox_2)
        self.label_4.setObjectName(u"label_4")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_4)

        self.begidxlineedit = QLineEdit(self.groupBox_2)
        self.begidxlineedit.setObjectName(u"begidxlineedit")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.begidxlineedit)

        self.label_5 = QLabel(self.groupBox_2)
        self.label_5.setObjectName(u"label_5")

        self.formLayout_2.setWidget(2, QFormLayout.LabelRole, self.label_5)

        self.endidxlineedit = QLineEdit(self.groupBox_2)
        self.endidxlineedit.setObjectName(u"endidxlineedit")

        self.formLayout_2.setWidget(2, QFormLayout.FieldRole, self.endidxlineedit)

        self.label_6 = QLabel(self.groupBox_2)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(3, QFormLayout.LabelRole, self.label_6)

        self.loctionlineedit = QLineEdit(self.groupBox_2)
        self.loctionlineedit.setObjectName(u"loctionlineedit")

        self.formLayout_2.setWidget(3, QFormLayout.FieldRole, self.loctionlineedit)


        self.horizontalLayout_2.addLayout(self.formLayout_2)


        self.horizontalLayout_4.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.widget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.horizontalLayout_3 = QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.label_7 = QLabel(self.groupBox_3)
        self.label_7.setObjectName(u"label_7")

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.label_7)

        self.instancevideokey = QLineEdit(self.groupBox_3)
        self.instancevideokey.setObjectName(u"instancevideokey")

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.instancevideokey)

        self.label_8 = QLabel(self.groupBox_3)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_3.setWidget(1, QFormLayout.LabelRole, self.label_8)

        self.instancebegidx = QLineEdit(self.groupBox_3)
        self.instancebegidx.setObjectName(u"instancebegidx")

        self.formLayout_3.setWidget(1, QFormLayout.FieldRole, self.instancebegidx)

        self.label_9 = QLabel(self.groupBox_3)
        self.label_9.setObjectName(u"label_9")

        self.formLayout_3.setWidget(2, QFormLayout.LabelRole, self.label_9)

        self.instanceendidx = QLineEdit(self.groupBox_3)
        self.instanceendidx.setObjectName(u"instanceendidx")

        self.formLayout_3.setWidget(2, QFormLayout.FieldRole, self.instanceendidx)

        self.label_10 = QLabel(self.groupBox_3)
        self.label_10.setObjectName(u"label_10")

        self.formLayout_3.setWidget(3, QFormLayout.LabelRole, self.label_10)

        self.instanceloction = QLineEdit(self.groupBox_3)
        self.instanceloction.setObjectName(u"instanceloction")

        self.formLayout_3.setWidget(3, QFormLayout.FieldRole, self.instanceloction)

        self.label_11 = QLabel(self.groupBox_3)
        self.label_11.setObjectName(u"label_11")

        self.formLayout_3.setWidget(4, QFormLayout.LabelRole, self.label_11)

        self.instancedis = QLineEdit(self.groupBox_3)
        self.instancedis.setObjectName(u"instancedis")

        self.formLayout_3.setWidget(4, QFormLayout.FieldRole, self.instancedis)


        self.horizontalLayout_3.addLayout(self.formLayout_3)


        self.horizontalLayout_4.addWidget(self.groupBox_3)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.nextinstance.setText(QCoreApplication.translate("Form", u"\u4e0b\u4e00\u4e2a\u5b9e\u4f8b", None))
        self.checkInstance.setText(QCoreApplication.translate("Form", u"\u6821\u9a8c\u8be5\u5b9e\u4f8b", None))
        self.checkAllInstances.setText(QCoreApplication.translate("Form", u"\u6821\u9a8c\u5b9e\u4f8b\u4eec", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Key", None))
        self.label.setText(QCoreApplication.translate("Form", u"word:", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"level:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"shapelet info", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"videokey:", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"begidx:", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"endidx:", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"loctions", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"instance info", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"videokey:", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"begidx:", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"endidx:", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"loctions", None))
        self.label_11.setText(QCoreApplication.translate("Form", u"distance:", None))
    # retranslateUi

