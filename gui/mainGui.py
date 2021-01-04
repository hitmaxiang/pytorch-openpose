'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-01-02 21:47:05
LastEditors: mario
LastEditTime: 2021-01-03 08:07:10
'''
import re
import os
import cv2
import sys
import h5py
import numpy as np
import configparser
import argparse

from copy import deepcopy
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import *
from DisplayThread import DisplayThread
from Annotation_Gui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化界面
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 初始化 Display Thread
        self.VideoThread = DisplayThread(worddictpath, subdictpath, videodir, recpoint)
        self.VideoThread.start()

        # 初始化界面的信号与功能函数
        self.InitWidgets()

    def InitWidgets(self):
        # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # 读取 标记文件
        recordfile = self.config.get('DEFAULT', 'recordfile')
        self.h5record = h5py.File(recordfile, mode='a')

        # wordlist 模块的配置

        # 初始化
        wordlistfile = self.config.get('DEFAULT', 'wordlistfile')
        self.ReadWordListFromFile(wordlistfile)
        # 搜索框： 信号-槽
        self.ui.wordexplore.textChanged.connect(self.FilterWordList)
        # 为 wordlist 设置右键函数
        self.ui.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.listWidget.customContextMenuRequested.connect(self.RightMenu_wordlist)
        
        # VideoPlayer 模块的初始化

        # Slider: 信号-槽
        self.ui.beginindex_spinBox.valueChanged.connect(self.UpdateLoopRange)
        self.ui.endindex_spinBox.valueChanged.connect(self.UpdateLoopRange)

        # 控制模块的初始化
        
        self.ui.btn_addneg.clicked.connect(lambda: self.AddAnnotation(-1))
        self.ui.btn_addpos.clicked.connect(lambda: self.AddAnnotation(1))
        self.ui.btn_nextsample.clicked.connect(lambda: self.BreakLoop(0))
        self.ui.btn_nextword.clicked.connect(lambda: self.BreakLoop(1))
        self.ui.btn_speedup.clicked.connect(lambda: self.SpeedCtrl(1))
        self.ui.btn_slowdown.clicked.connect(lambda: self.SpeedCtrl(-1))

        # VideoThread： 信号-槽

        self.VideoThread.ImgDisplaySignal.connect(self.UpdateVideoPlayerParames)
        self.VideoThread.StatesOutSignal.connect(self.UpdateDisplayparams)
    
    @Slot()
    def ReadWordListFromFile(self, file):
        wordlist = []
        if os.path.exists(file):
            with open(file, 'r', encoding='utf8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line != '':
                    wordlist.append(line)
        wordlist.sort()
        self.DisplayWordList(wordlist)
        self.wordlist = wordlist
        
    @Slot()
    def FilterWordList(self, pattern):
        pattern = r'%s' % pattern
        templist = []
        for word in self.wordlist:
            if re.search(pattern, word, flags=re.IGNORECASE):
                templist.append(word)
        self.DisplayWordList(templist)
    
    @Slot()
    def RightMenu_wordlist(self, pos):
        word = self.ui.listWidget.itemAt(pos).text()
        menu = QMenu()
        menu.addAction('samples').triggered.connect(lambda: self.DemonWord('sample', word))
        menu.addSeparator()
        menu.addAction('shapelets').triggered.connect(lambda: self.DemonWord('shapelet', word))
        menu.exec_(self.ui.listWidget.mapToGlobal(pos))

    def DisplayWordList(self, wordlist):
        self.ui.listWidget.clear()
        self.ui.listWidget.addItems(wordlist)

    def DemonWord(self, mode, word):
        print('want to demonstrate word: %s in mode:%s' % (word, mode))
        self.VideoThread.UpdateWord(word, mode)

    @Slot()
    def UpdateVideoPlayerParames(self, currentidx, speed):
        self.ui.currentframe.setText(str(currentidx))
        self.ui.speed_text.setText('x %.02f' % speed)
        # update the img
        img = deepcopy(self.VideoThread.shareImg)

        height, width, depth = img.shape
        wb, hb = self.ui.videolabel.width(), self.ui.videolabel.height()
        f = min(wb/width, hb/height)
        img = cv2.resize(img, (0, 0), fx=f, fy=f)
        height, width, depth = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, width, height, width*depth, QImage.Format_RGB888)
        self.ui.videolabel.clear()
        self.ui.videolabel.setPixmap(QPixmap.fromImage(img))

    @Slot()
    def UpdateDisplayparams(self, word, videonum, offset, length, speed, demoindex):
        self.ui.word_text.setText(word)
        self.ui.videokey_text.setText(videonum)
        self.ui.offset_text.setText(offset)
        self.ui.length_text.setText(length)
        self.ui.idx_text.setText(demoindex)

        length = int(length) - 1
        self.ui.beginindex_Slider.setMaximum(length)
        self.ui.beginindex_spinBox.setMaximum(length)

        self.ui.endindex_Slider.setMaximum(length)
        self.ui.endindex_spinBox.setMaximum(length)
        
        self.ui.beginindex_spinBox.setValue(0)
        self.ui.endindex_spinBox.setValue(length)

        h5key = '%s/%s/%s' % (word, videonum, offset)
        if h5key in self.h5record.keys():
            self.ui.annotated_text.setText('True')
            self.ui.postive_label.setVisible(True)
            self.ui.positive_text.setVisible(True)
            data = self.h5record[h5key][:]
            if data[0] == 1:
                self.ui.positive_text.setText('True')

                self.ui.beginidx_label.setVisible(True)
                self.ui.beginidx_text.setVisible(True)
                self.ui.endidx_label.setVisible(True)
                self.ui.endidx_text.setVisible(True)

                self.ui.beginidx_text.setText(str(int(data[1])))
                self.ui.endidx_text.setText(str(int(data[2])))
                self.ui.beginindex_spinBox.setValue(int(data[1]))
                self.ui.endindex_spinBox.setValue(int(data[2]))
            else:
                self.ui.positive_text.setText('False')

                self.ui.beginidx_label.setVisible(False)
                self.ui.beginidx_text.setVisible(False)
                self.ui.endidx_label.setVisible(False)
                self.ui.endidx_text.setVisible(False)
        else:
            self.ui.annotated_text.setText('False')
            self.ui.postive_label.setVisible(False)
            self.ui.positive_text.setVisible(False)
            self.ui.beginidx_label.setVisible(False)
            self.ui.beginidx_text.setVisible(False)
            self.ui.endidx_label.setVisible(False)
            self.ui.endidx_text.setVisible(False)
            
    @Slot()
    def UpdateLoopRange(self):
        a = self.ui.beginindex_spinBox.value()
        b = self.ui.endindex_spinBox.value()
        self.VideoThread.UpdateLoopRange(min(a, b), max(a, b))
    
    @Slot()
    def AddAnnotation(self, flag):
        word = self.ui.word_text.text()
        videokey = self.ui.videokey_text.text()
        offset = self.ui.offset_text.text()

        if word == '' or videokey == '' or offset == '':
            print('the annotation info is not completed')
            return

        h5key = '%s/%s/%s' % (word, videokey, offset)
        if h5key in self.h5record.keys():
            msgBox = QMessageBox()
            msgBox.setText('This sample is already annotated!!!')
            msgBox.setInformativeText('Do you want to overwrite the exist annotation?')
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            ret = msgBox.exec_()
            if ret != QMessageBox.Yes:
                return
        else:
            self.h5record.create_dataset(h5key, (3,))
        
        if flag == -1:
            self.h5record[h5key][:] = np.array([-1, -1, -1])
        else:
            beginidx = self.ui.beginindex_spinBox.value()
            endidx = self.ui.endindex_spinBox.value()
            minv, maxv = min(beginidx, endidx), max(beginidx, endidx)
            self.h5record[h5key][:] = np.array([1, minv, maxv])
            
        self.h5record.flush()
        self.VideoThread.NextSample()

    @Slot()
    def BreakLoop(self, level):
        if level == 0:
            self.VideoThread.NextSample()
        elif level == 1:
            self.VideoThread.NextWord()
    
    @Slot()
    def SpeedCtrl(self, orien):
        if orien == 1:
            self.VideoThread.speed = min(self.VideoThread.speed+0.1, 5)
        elif orien == -1:
            self.VideoThread.speed = max(self.VideoThread.speed/2, 0.05)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', action='store_true')

    args = parser.parse_args()
    server = args.server

    if server is True:
        videodir = '/home/mario/signdata/spbsl/normal'
    else:
        videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'

    worddictpath = '../data/spbsl/WordDict.pkl'
    subdictpath = '../data/spbsl/SubtitleDict.pkl'
    recpoint = [(700, 100), (1280, 720)]
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())