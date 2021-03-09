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
        Form.resize(967, 235)
        self.horizontalLayout_6 = QHBoxLayout(Form)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.groupBox = QGroupBox(Form)
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

        self.groupBox_2 = QGroupBox(Form)
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

        self.label_12 = QLabel(self.groupBox_2)
        self.label_12.setObjectName(u"label_12")

        self.formLayout_2.setWidget(4, QFormLayout.LabelRole, self.label_12)

        self.shapeletindex = QLineEdit(self.groupBox_2)
        self.shapeletindex.setObjectName(u"shapeletindex")

        self.formLayout_2.setWidget(4, QFormLayout.FieldRole, self.shapeletindex)


        self.horizontalLayout_2.addLayout(self.formLayout_2)


        self.horizontalLayout_4.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(Form)
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


        self.horizontalLayout_6.addLayout(self.horizontalLayout_4)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.curinstanceindex = QSpinBox(Form)
        self.curinstanceindex.setObjectName(u"curinstanceindex")

        self.horizontalLayout_5.addWidget(self.curinstanceindex)

        self.posnumedit = QLineEdit(Form)
        self.posnumedit.setObjectName(u"posnumedit")

        self.horizontalLayout_5.addWidget(self.posnumedit)


        self.verticalLayout.addLayout(self.horizontalLayout_5)

        self.nextinstance = QPushButton(Form)
        self.nextinstance.setObjectName(u"nextinstance")

        self.verticalLayout.addWidget(self.nextinstance)

        self.checkInstance = QPushButton(Form)
        self.checkInstance.setObjectName(u"checkInstance")

        self.verticalLayout.addWidget(self.checkInstance)

        self.checkAllInstances = QPushButton(Form)
        self.checkAllInstances.setObjectName(u"checkAllInstances")

        self.verticalLayout.addWidget(self.checkAllInstances)


        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.horizontalLayout_6.addLayout(self.verticalLayout_2)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Form", None))
        self.groupBox.setTitle(QCoreApplication.translate("Form", u"Key", None))
        self.label.setText(QCoreApplication.translate("Form", u"word:", None))
        self.label_2.setText(QCoreApplication.translate("Form", u"level:", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("Form", u"shapelet info", None))
        self.label_3.setText(QCoreApplication.translate("Form", u"videokey:", None))
        self.label_4.setText(QCoreApplication.translate("Form", u"begidx:", None))
        self.label_5.setText(QCoreApplication.translate("Form", u"endidx:", None))
        self.label_6.setText(QCoreApplication.translate("Form", u"loctions:", None))
        self.label_12.setText(QCoreApplication.translate("Form", u"Index:", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("Form", u"instance info", None))
        self.label_7.setText(QCoreApplication.translate("Form", u"videokey:", None))
        self.label_8.setText(QCoreApplication.translate("Form", u"begidx:", None))
        self.label_9.setText(QCoreApplication.translate("Form", u"endidx:", None))
        self.label_10.setText(QCoreApplication.translate("Form", u"loctions", None))
        self.label_11.setText(QCoreApplication.translate("Form", u"distance:", None))
        self.nextinstance.setText(QCoreApplication.translate("Form", u"\u4e0b\u4e00\u4e2a\u5b9e\u4f8b", None))
        self.checkInstance.setText(QCoreApplication.translate("Form", u"\u6821\u9a8c\u8be5\u5b9e\u4f8b", None))
        self.checkAllInstances.setText(QCoreApplication.translate("Form", u"\u6821\u9a8c\u5b9e\u4f8b\u4eec", None))
    # retranslateUi

