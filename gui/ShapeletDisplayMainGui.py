'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-03-08 16:14:15
LastEditors: mario
LastEditTime: 2021-03-09 15:45:16
'''
import sys
sys.path.append('../')
import os
import re
import h5py
import numpy as np
from srcmx import utilmx
from srcmx import PreprocessingData as PD

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from Ui_DisplayShapeletRecord import Ui_Form as Ui_Shaplelet
from Ui_MainWindow import Ui_MainWindow 


class ShapeletDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.Display = Ui_Shaplelet()
        self.Display.setupUi(self)

        self.motionfilepath = '../data/spbsl/motiondata.hdf5'
        self.shapeletfilepath = None
        self.Index = 0

        self.InitWidget()

    def InitWidget(self):
        self.Display.nextinstance.clicked.connect(self.NextInstance)
        self.Display.checkInstance.clicked.connect(self.CheckInstance)
        self.Display.checkAllInstances.clicked.connect(self.CheckAllInstance)
        self.Display.curinstanceindex.valueChanged.connect(self.UpdateFromSpin)
        self.Display.wordcomboBox.currentIndexChanged.connect(self.UpdateDisplayInfo)
        self.Display.levelcomboBox.currentIndexChanged.connect(self.UpdateDisplayInfo)
    
    def UpdateFromSpin(self):
        self.Index = self.Display.curinstanceindex.value()
        self.UpdateDisplayInfo()
        
    def CheckAllInstance(self):
        if self.shapeletfilepath is None or (not os.path.exists(self.shapeletfilepath)):
            return
        word = self.Display.wordcomboBox.currentText()
        level = self.Display.levelcomboBox.currentText()
        
        with h5py.File(self.shapeletfilepath, 'r') as h5file:
            N = len(h5file[word][level]['dists'][:])
        if N > 0:
            for i in range(N):
                self.Index = i
                self.UpdateDisplayInfo()
                self.CheckInstance()
    
    def NextInstance(self):
        N = self.Display.posnumedit.text()
        if N is not None and int(N) > 0:
            self.Index = (self.Index + 1) % int(N)
            self.UpdateDisplayInfo()
    
    def CheckInstance(self):
        word = self.Display.wordcomboBox.currentText()
        level = self.Display.levelcomboBox.currentText()
        if word is None or level is None or self.shapeletfilepath is None:
            return
        
        with h5py.File(self.shapeletfilepath, 'r') as h5file:
            infopattern = r'fx:(.+)-datamode:(.+)-featuremode:(\d+)$'
            info = utilmx.Encode(h5file[word]['loginfo'][0])
            fx, datamode, featuremode = re.findall(infopattern, info)[0]
            featuremode = int(featuremode)
            
            begidx = int(self.Display.begidxlineedit.text())
            endidx = int(self.Display.endidxlineedit.text())
            loc = int(self.Display.loctionlineedit.text())
            vdokey = self.Display.videokeylineedit.text()
            
            in_begidx = int(self.Display.instancebegidx.text())
            in_endidx = int(self.Display.instanceendidx.text())
            in_loc = int(self.Display.instanceloction.text())
            in_vdokey = self.Display.instancevideokey.text()
            dist = float(self.Display.instancedis.text())
        
        with h5py.File(self.motionfilepath, 'r') as motionfile:

            posedata = motionfile['posedata/pose/%s' % vdokey][begidx:endidx].astype(np.float32)
            handdata = motionfile['handdata/hand/%s' % vdokey][begidx:endidx].astype(np.float32)
            shapeletdata = np.concatenate((posedata, handdata), axis=1)
            shapeletdata = shapeletdata[loc:loc+int(level)]
            shapeletdata = PD.MotionJointFeatures(shapeletdata, datamode, featuremode)
            shapeletdata = np.reshape(shapeletdata, (shapeletdata.shape[0], -1))
            # shapeletdata = shapeletdata[loc:loc+int(level)]

            posedata = motionfile['posedata/pose/%s' % in_vdokey][in_begidx:in_endidx].astype(np.float32)
            handdata = motionfile['handdata/hand/%s' % in_vdokey][in_begidx:in_endidx].astype(np.float32)
            Instancedata = np.concatenate((posedata, handdata), axis=1)

            Instancedata = PD.MotionJointFeatures(Instancedata, datamode, featuremode)
            Instancedata = np.reshape(Instancedata, (Instancedata.shape[0], -1))

            dists = utilmx.SlidingDistance(shapeletdata, Instancedata)
            # print(dists)
            idx = np.argmin(dists)
            print('%d==>record:%d-%f, now:%d-%f' % (self.Index, in_loc, dist, idx, min(dists)))
    
    def UpdateDisplayInfo(self):
        if self.shapeletfilepath is None or (not os.path.exists(self.shapeletfilepath)):
            return
        word = self.Display.wordcomboBox.currentText()
        level = self.Display.levelcomboBox.currentText()

        if word == '' or level == '':
            return
        
        with h5py.File(self.shapeletfilepath, 'r') as h5file:
            if '%s/%s' % (word, level) in h5file.keys():
                Idx = h5file[word][level]['shapelet'][:]
                if len(Idx) != 1:
                    print('the shapeletfile should from the shapelet finding')
                    return

                Idx = Idx[0]
                begidx, endidx = h5file[word]['sampleidxs'][Idx]
                videokey = utilmx.Encode(h5file[word]['videokeys'][Idx])
                location = h5file[word][level]['locs'][Idx]
                self.Display.videokeylineedit.setText(videokey)
                self.Display.begidxlineedit.setText(str(begidx))
                self.Display.endidxlineedit.setText(str(endidx))
                self.Display.loctionlineedit.setText(str(location))

                begidx, endidx = h5file[word]['sampleidxs'][self.Index]
                videokey = utilmx.Encode(h5file[word]['videokeys'][self.Index])
                dist = h5file[word][level]['dists'][self.Index]
                location = h5file[word][level]['locs'][self.Index]

                self.Display.instancevideokey.setText(videokey)
                self.Display.instancebegidx.setText(str(begidx))
                self.Display.instanceendidx.setText(str(endidx))
                self.Display.instanceloction.setText(str(location))
                self.Display.instancedis.setText(str(dist))

                N = len(h5file[word]['sampleidxs'][:])
                self.Display.posnumedit.setText(str(N))
                self.Display.curinstanceindex.setRange(0, N-1)
                self.Display.curinstanceindex.setValue(self.Index)
                self.Display.shapeletindex.setText(str(Idx))

    def OpenFile(self, filepath):
        if not os.path.isfile(filepath):
            return
        self.shapeletfilepath = filepath

        with h5py.File(filepath, 'r') as h5file:
            words = list(h5file.keys())
            self.Display.wordcomboBox.clear()
            for word in words:
                self.Display.wordcomboBox.addItem(word)
            self.Display.wordcomboBox.setCurrentIndex(0)
            self.Display.levelcomboBox.clear()
            for levelkey in h5file[word].keys():
                if re.match(r'^\d+$', levelkey) is not None:
                    self.Display.levelcomboBox.addItem(levelkey)
            self.Display.levelcomboBox.setCurrentIndex(0)
        self.Index = 0
        self.UpdateDisplayInfo()


class MainGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.shapeletDisplay = ShapeletDisplay()
        
        self.mainwindow = Ui_MainWindow()
        self.mainwindow.setupUi(self)

        self.InitWidgets()
        self.setCentralWidget(self.shapeletDisplay)

    def InitWidgets(self):
        self.mainwindow.actionShaperecord.triggered.connect(self.ChooseRecordFile)
        # self.shapeletDisplay.OpenFile('../data/spbsl/shapeletED.hdf5')
        self.shapeletDisplay.OpenFile('../data/spbsl/bk1_shapeletED.hdf5')

    
    def ChooseRecordFile(self):
        results = QFileDialog.getOpenFileName(self,
                                              "Open recordfile", 
                                              "../data/spbsl",
                                              "record Files (*.hdf5)")
        
        filepath = results[0]
        if os.path.exists(filepath):
            self.shapeletDisplay.OpenFile(filepath)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec_()) 