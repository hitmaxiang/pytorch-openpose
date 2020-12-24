'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-22 13:51:11
LastEditors: mario
LastEditTime: 2020-12-24 21:23:00
'''
import cv2
import re, os, sys
import numpy as np
import DisplayThread as DH
from DisplayThread import DisplayThread
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtCore import Slot, Qt, Signal
from PySide2.QtGui import QPicture, QImage, QPixmap
from PySide2.QtWidgets import *


class WordListDisplay(QWidget):
    ChooseWordSignal = Signal(str, str)

    def __init__(self, input=None):
        super().__init__()
        self.wordlist = self.ReadfromFile(input)

        self.wordexplore = QLineEdit()
        self.listwidget = QListWidget()

        self.vlayout = QVBoxLayout()
        self.vlayout.addWidget(self.wordexplore)
        self.vlayout.addWidget(self.listwidget)
        self.setLayout(self.vlayout)

        self.wordexplore.textChanged[str].connect(self.update)
        self.listwidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.listwidget.customContextMenuRequested.connect(self.rightMenuShow)

        self.fill_list(self.wordlist)
    
    def ReadfromFile(self, file):
        wordlist = []
        if os.path.exists(file):
            with open(file, 'r', encoding='utf8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line != '':
                    wordlist.append(line)
        wordlist.sort()
        
        return wordlist
    
    def fill_list(self, wordlist):
        self.listwidget.clear()
        self.listwidget.addItems(wordlist)
    
    @Slot()
    def update(self, pattern):
        pattern = r'%s' % pattern
        templist = []
        for word in self.wordlist:
            if re.search(pattern, word, flags=re.IGNORECASE):
                templist.append(word)
        self.fill_list(templist)

    @Slot()
    def rightMenuShow(self, pos):
        word = self.listwidget.itemAt(pos).text()
        menu = QMenu()
        menu.addAction('samples').triggered.connect(lambda: self.demons('sample', word))
        menu.addSeparator()
        menu.addAction('shapelets').triggered.connect(lambda: self.demons('shapelet', word))
        menu.exec_(self.listwidget.mapToGlobal(pos))
    
    @Slot()
    def demons(self, mode, word):
        print('want to demonstrate word: %s in mode:%s' % (word, mode))
        self.ChooseWordSignal.emit(word, mode)


class VideoPlayer(QWidget):
    LoopIndexsignal = Signal(int, int)

    def __init__(self, shape=None):
        super().__init__()
        # 起始帧显示
        beginlayout = QHBoxLayout()
        beginlayout.addWidget(QLabel('Start index:'))
        self.startindex = QLineEdit()
        beginlayout.addWidget(self.startindex)
        beginlayout.addStretch(1)

        # 起始帧滑动条
        self.Beginslider = QSlider(Qt.Horizontal)
        
        # 终止帧显示
        endlayout = QHBoxLayout()
        endlayout.addWidget(QLabel('End index:'))
        self.endindex = QLineEdit()
        endlayout.addWidget(self.endindex)
        endlayout.addStretch(1)
        
        # current index
        self.speedupbtn = QPushButton('SpeedUp')
        self.slowdownbtn = QPushButton('SlowDown')
        displaylayout = QHBoxLayout()
        displaylayout.addWidget(self.speedupbtn)
        displaylayout.addStretch(1)
        displaylayout.addWidget(QLabel('CurrentFrame：'))
        self.currentindex = QLineEdit()
        displaylayout.addWidget(self.currentindex)
        displaylayout.addStretch(1)
        displaylayout.addWidget(self.slowdownbtn)

        # 终止帧滑动条
        self.Endslider = QSlider(Qt.Horizontal)

        # 图片显示区域
        self.videoframe = QLabel('video')

        # 加载布局
        vlayout = QVBoxLayout()
        vlayout.addLayout(beginlayout)
        vlayout.addWidget(self.Beginslider)
        vlayout.addSpacing(5)
        vlayout.addLayout(endlayout)
        vlayout.addWidget(self.Endslider)
        vlayout.addSpacing(5)
        vlayout.addLayout(displaylayout)
        vlayout.addSpacing(10)
        vlayout.addWidget(self.videoframe)

        self.setLayout(vlayout)

        # signal and slot

        self.startindex.editingFinished.connect(self.Startindex2Slider)
        self.Beginslider.valueChanged.connect(self.Slider2Startindex)
        self.endindex.editingFinished.connect(self.Endindex2Slider)
        self.Endslider.valueChanged.connect(self.Slider2Endindex)
        
        self.parametersconfig()
    
    @Slot()
    def Startindex2Slider(self):
        try:
            s = self.startindex.text()
            value = int(s)
            if not (self.Beginslider.maximum() >= value >= self.Beginslider.minimum()):
                self.Slider2Startindex()
            else:
                self.Beginslider.setValue(value)
        except Exception:
            self.Slider2Startindex()
    
    @Slot()
    def Endindex2Slider(self):
        try:
            s = self.endindex.text()
            value = int(s)
            if not (self.Endslider.maximum() >= value >= self.Endslider.minimum()):
                self.Slider2Endindex()
            else:
                self.Endslider.setValue(value)
        except Exception:
            self.Slider2Endindex()
    
    @Slot()
    def Slider2Endindex(self):
        value = self.Endslider.value()
        self.endindex.setText(str(value))
        # other thing
        beginvalue = self.Beginslider.value()
        self.LoopIndexsignal.emit(min(value, beginvalue), max(value, beginvalue))
    
    @Slot()
    def Slider2Startindex(self):
        value = self.Beginslider.value()
        self.startindex.setText(str(value))
        # other thing
        endvalue = self.Endslider.value()
        self.LoopIndexsignal.emit(min(value, endvalue), max(value, endvalue))

    @Slot()
    def parametersconfig(self, shape=(320, 320), length=100):
        self.videoframe.setMinimumSize(320, 320)
        self.videoframe.setFixedSize(580, 620)
        # self.videoframe.setAttribute(Qt.WA_NoBackground)
        # self.videoframe.setAttribute(Qt.WA_OpaquePaintEvent)
        # self.videoframe.setAttribute(Qt.WA_UpdatesDisabled)
        # self.videoframe.setBackgroundRole()
        # self.videoframe.setScaledContents(True)
        self.startindex.setValidator(QtGui.QIntValidator())
        self.endindex.setValidator(QtGui.QIntValidator())
        self.Beginslider.setMaximum(length)
        self.Endslider.setMaximum(length)

        self.currentindex.setReadOnly(True)
    
    @Slot()
    def DisplayImg(self, currentindex, img=None):
        if img is None:
            # img = np.random.randint(0, 255, size=(580, 620, 3), dtype=np.uint8)
            img = DH.ShareImg[0]
            # cv2.imshow('demo', img)
            
        height, width, depth = img.shape
        # cv2.imshow('demo', img)
        
        self.currentindex.setText(str(currentindex))
        Qimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Qimg = QImage(Qimg.data, width, height, width*depth, QImage.Format_RGB888)
        self.videoframe.clear()
        self.videoframe.setPixmap(QPixmap.fromImage(Qimg))
        # self.videoframe.update()
    
    @Slot()
    def updateparameters(self, length):
        self.Beginslider.setMinimum(0)
        self.Beginslider.setMaximum(length)
        self.Beginslider.setValue(0)
        
        self.Endslider.setMinimum(0)
        self.Endslider.setMaximum(length)
        self.Endslider.setValue(length)


class StatusDisplay(QWidget):
    def __init__(self):
        super().__init__()

        # 构建显示控件
        self.wordstr = QLineEdit('')
        self.videokeynum = QLineEdit('')
        self.offset = QLineEdit('')
        self.length = QLineEdit('')
        self.speed = QLineEdit('')
        self.demoindex = QLineEdit('')

        # grid 布局
        gridlayout = QGridLayout()

        gridlayout.addWidget(QLabel('word:'), 0, 0)
        gridlayout.addWidget(self.wordstr, 0, 1)
        gridlayout.addWidget(QLabel('videonum:'), 0, 2)
        gridlayout.addWidget(self.videokeynum, 0, 3)

        gridlayout.addWidget(QLabel('offset:'), 1, 0)
        gridlayout.addWidget(self.offset, 1, 1)
        gridlayout.addWidget(QLabel('length:'), 1, 2)
        gridlayout.addWidget(self.length, 1, 3)

        gridlayout.addWidget(QLabel('speend:'), 2, 0)
        gridlayout.addWidget(self.speed, 2, 1)
        gridlayout.addWidget(QLabel('demoindex:'), 2, 2)
        gridlayout.addWidget(self.demoindex, 2, 3)

        hlayout = QHBoxLayout()
        self.add_annotation = QPushButton('Add Annotation')
        self.nextsample = QPushButton('Next sample')
        hlayout.addWidget(self.add_annotation)
        hlayout.addWidget(self.nextsample)

        vlayout = QVBoxLayout()
        vlayout.addLayout(gridlayout)
        vlayout.addStretch(1)
        vlayout.addLayout(hlayout)

        self.setLayout(vlayout)

    @Slot()
    def SetStatus(self, word, videonum, offset, length, speed, demoindex):
        self.wordstr.setText(word)
        self.videokeynum.setText(videonum)
        self.offset.setText(str(offset))
        self.length.setText(str(length))
        self.speed.setText(str(speed))
        self.demoindex.setText(demoindex)
    
    
class DisPlayFrame(QWidget):
    def __init__(self, inputname):
        super().__init__()

        hlayout = QHBoxLayout()
        self.wordindex = WordListDisplay(inputname)
        self.videoplayer = VideoPlayer()
        self.statusdisplay = StatusDisplay()

        hlayout.addWidget(self.wordindex)
        hlayout.addWidget(self.videoplayer)
        hlayout.addWidget(self.statusdisplay)
        self.setLayout(hlayout)
    

class MainGui(QMainWindow):
    def __init__(self, worddictpath, subdictpath, videodir, recpoint):
        super().__init__()
        # 初始化显示控件
        self.setWindowTitle('Annotation & Display system')
        self.displaygui = DisPlayFrame('./wordlist.txt')
        self.setCentralWidget(self.displaygui)

        # 初始化显示线程控件
        self.displaythread = DisplayThread(worddictpath, subdictpath, videodir, recpoint)
        self.displaythread.start()

        global shareImg
        shareImg = [0]
        shareImg[0] = self.displaythread.shareImg

        self.signalInit()

    def signalInit(self):
        self.displaythread.StatesOutSignal.connect(self.displaygui.statusdisplay.SetStatus)
        self.displaythread.ImgDisplaySignal.connect(self.CopyImg)
        self.displaythread.DisplayLengthSig.connect(self.displaygui.videoplayer.updateparameters)

        self.displaygui.wordindex.ChooseWordSignal.connect(self.displaythread.UpdateWord)
        self.displaygui.videoplayer.LoopIndexsignal.connect(self.displaythread.UpdateLoopRange)
        self.displaygui.statusdisplay.add_annotation.clicked.connect(self.add_annotation_record)
        self.displaygui.statusdisplay.nextsample.clicked.connect(self.nextsamplefunc)
        self.displaygui.videoplayer.speedupbtn.clicked.connect(self.speedupfunc)
        self.displaygui.videoplayer.slowdownbtn.clicked.connect(self.slowdownfunc)
        
    @Slot()
    def CopyImg(self, index, speed):
        pass
        img = self.displaythread.shareImg
        self.displaygui.videoplayer.DisplayImg(index, img)
        self.displaygui.statusdisplay.speed.setText('x %.2f' % speed)
    
    @Slot()
    def add_annotation_record(self):
        word = self.displaygui.statusdisplay.wordstr.text()
        videokeynum = self.displaygui.statusdisplay.videokeynum.text()
        offset = self.displaygui.statusdisplay.offset.text()
        beginindex = self.displaygui.videoplayer.Beginslider.value()
        endindex = self.displaygui.videoplayer.Endslider.value()

        bi = min(beginindex, endindex)
        ei = max(beginindex, endindex)

        with open('./annotation.rec', 'a', encoding='utf8') as f:
            string = '\n%s\t%s\t%s\t%d\t%d' % (word, videokeynum, offset, bi, ei)
            f.write(string)
    
    @Slot()
    def nextsamplefunc(self):
        self.displaythread.displayloop = False
    
    @Slot()
    def speedupfunc(self):
        self.displaythread.speed = min(self.displaythread.speed+0.1, 5)
    
    @Slot()
    def slowdownfunc(self):
        self.displaythread.speed = max(self.displaythread.speed/2, 0.05)


if __name__ == "__main__":
    worddictpath = '../data/spbsl/WordDict.pkl'
    subdictpath = '../data/spbsl/SubtitleDict.pkl'
    videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'
    Recpoint = [(700, 100), (1280, 720)]
    app = QApplication([])
    GUI = MainGui(worddictpath, subdictpath, videodir, Recpoint)
    GUI.show()
    sys.exit(app.exec_())


