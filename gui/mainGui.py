'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-01-02 21:47:05
LastEditors: mario
LastEditTime: 2021-01-07 22:34:12
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
        # menuBar
        self.ui.actionrecordfile.triggered.connect(self.ChooseRecordFile)
        # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # 读取 标记文件
        recordfile = self.config.get('DEFAULT', 'recordfile')
        self.h5annotation = h5py.File(recordfile, mode='a')

        # wordlist 模块的配置

        # 初始化
        wordlistfile = self.config.get('DEFAULT', 'wordlistfile')
        self.ReadWordListFromFile(wordlistfile)
        # 搜索框： 信号-槽
        self.ui.wordexplore.textChanged.connect(self.FilterWordList)
        self.ui.wordexplore.returnPressed.connect(self.WordEntered)
        # 为 wordlist 设置右键函数
        self.ui.listWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.listWidget.customContextMenuRequested.connect(self.RightMenu_wordlist)

        # 设置 单击和双击 函数
        self.ui.listWidget.itemClicked.connect(self.DisplaySelectedWord)
        self.ui.listWidget.itemDoubleClicked.connect(self.DemoSelectedWord)
        
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

        # Shapelet 模块的初始化
        self.ui.m_len_combo.currentTextChanged.connect(self.VideoThread.SetCurrent_m_len)

        # VideoThread： 信号-槽

        self.VideoThread.ImgDisplaySignal.connect(self.UpdateVideoPlayerParames)
        self.VideoThread.StatesOutSignal.connect(self.UpdateDisplayparams)
        self.VideoThread.SampleInfoSignal.connect(self.UpdateSampleInfo)
    
    @Slot()
    def WordEntered(self):
        word = self.ui.wordexplore.text()
        if word not in self.wordlist:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText('输入提示?')
            msgBox.setInformativeText('列表中没有该单词, 确认进行展示吗?')
            msgBox.addButton(QMessageBox.Cancel)
            accept_btn = msgBox.addButton('展示', QMessageBox.ActionRole)
            added_btn = msgBox.addButton('加入列表', QMessageBox.ActionRole)
            all_btn = msgBox.addButton('加入+展示', QMessageBox.ActionRole)
            
            msgBox.exec_()
            clickbtn = msgBox.clickedButton()
            if clickbtn == accept_btn:
                self.DemonWord('samples', word)
            elif clickbtn == added_btn:
                self.wordlist.insert(0, word)
                self.DisplayWordList(self.wordlist)
            elif clickbtn == all_btn:
                self.wordlist.insert(0, word)
                self.DisplayWordList(self.wordlist)
                self.DemonWord('samples', word)

    @Slot()
    def DemoSelectedWord(self, item):
        word = item.text()
        self.DemonWord('sample', word)

    @Slot()
    def DisplaySelectedWord(self, item):
        word = item.text()
        count = self.ui.listWidget.count()
        cur_row = self.ui.listWidget.currentRow()

        pos_num, neg_num = 0, 0
        if word in self.h5annotation.keys():
            for videokey in self.h5annotation[word].keys():
                for offset in self.h5annotation[word][videokey]:
                    data = self.h5annotation[word][videokey][offset]
                    if data[0] == 1:
                        pos_num += 1
                    else:
                        neg_num += 1
        
        annoinfo = '%s pos:%d/neg: %d [%d/%d]' % (word, pos_num, neg_num, cur_row, count)
        self.ui.statusbar.showMessage(annoinfo)

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
        self.allwordlist = wordlist
        self.wordlist = wordlist
        self.VideoThread.wordlist = wordlist
        
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
        item = self.ui.listWidget.itemAt(pos)
        menu = QMenu()
        if item is not None:
            word = self.ui.listWidget.itemAt(pos).text()
            menu.addAction('samples').triggered.connect(lambda: self.DemonWord('sample', word))
            menu.addAction('shapelets').triggered.connect(lambda: self.DemonWord('shapelet', word))
            menu.addAction('delAnntation').triggered.connect(lambda: self.DeleteAnnotation(word))
            menu.addSeparator()

        menu.addAction('annotationlist').triggered.connect(lambda: self.WordListShow(1))
        menu.addAction('allwordlist').triggered.connect(lambda: self.WordListShow(0))
        menu.exec_(self.ui.listWidget.mapToGlobal(pos))
    
    @Slot()
    def WordListShow(self, mode):
        if mode == 0:
            self.wordlist = self.allwordlist
        elif mode == 1:
            # only show annotated word
            tempwordlist = []
            for word in list(self.h5annotation.keys()):
                tempwordlist.append(word)
            self.wordlist = tempwordlist
        
        self.DisplayWordList(self.wordlist)
    
    @Slot()
    def DeleteAnnotation(self, word):
        msgBox = QMessageBox.question
        if word in self.h5annotation.keys():
            num = len(list(self.h5annotation[word].keys()))
            ret = msgBox(self, '警告', '%s的%d个标记将被删除，是否确认' % (word, num),
                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret == QMessageBox.Yes:
                del self.h5annotation[word]
                self.h5annotation.flush()
        else:
            msgBox(self, '提示', '%s 不在标记文件中！' % word, QMessageBox.Ok, QMessageBox.Ok)

    def DisplayWordList(self, wordlist):
        self.ui.listWidget.clear()
        self.ui.listWidget.addItems(wordlist)

    def DemonWord(self, mode, word):
        print('want to demonstrate word: %s in mode:%s' % (word, mode))
        self.VideoThread.UpdateWord(word, mode)

    @Slot()
    def UpdateVideoPlayerParames(self, currentidx, speed, shapelet_begin):
        self.ui.currentframe.setText(str(currentidx))
        self.ui.speed_text.setText('x %.02f' % speed)
        self.ui.location_text.setText('%d' % shapelet_begin)

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

        # update the shapelet info
        img = deepcopy(self.VideoThread.ShapeletImg)
        height, width, depth = img.shape
        wb, hb = self.ui.shapelet_video.width(), self.ui.shapelet_video.height()
        f = min(wb/width, hb/height)
        img = cv2.resize(img, (0, 0), fx=f, fy=f)
        height, width, depth = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, width, height, width*depth, QImage.Format_RGB888)
        self.ui.shapelet_video.clear()
        self.ui.shapelet_video.setPixmap(QPixmap.fromImage(img))

    @Slot()
    def UpdateDisplayparams(self, word, videonum, offset, length, index):
        self.ui.word_text.setText(word)
        self.ui.videokey_text.setText(videonum)
        self.ui.offset_text.setText(str(offset))
        self.ui.length_text.setText(str(length))
        self.ui.idx_text.setText('%d/%d' % (index+1, self.Count))

        length = int(length) - 1
        self.ui.beginindex_Slider.setMaximum(length)
        self.ui.beginindex_spinBox.setMaximum(length)

        self.ui.endindex_Slider.setMaximum(length)
        self.ui.endindex_spinBox.setMaximum(length)
        
        self.ui.beginindex_spinBox.setValue(0)
        self.ui.endindex_spinBox.setValue(length)

        h5key = '%s/%s/%s' % (word, videonum, offset)
        if h5key in self.h5annotation.keys():
            self.ui.annotated_text.setText('True')
            self.ui.postive_label.setVisible(True)
            self.ui.positive_text.setVisible(True)
            data = self.h5annotation[h5key][:]
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
        if h5key in self.h5annotation.keys():
            msgBox = QMessageBox()
            msgBox.setText('This sample is already annotated!!!')
            msgBox.setInformativeText('Do you want to overwrite the exist annotation?')
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            ret = msgBox.exec_()
            if ret != QMessageBox.Yes:
                return
        else:
            self.h5annotation.create_dataset(h5key, (3,))
        
        if flag == -1:
            self.h5annotation[h5key][:] = np.array([-1, -1, -1])
        else:
            beginidx = self.ui.beginindex_spinBox.value()
            endidx = self.ui.endindex_spinBox.value()
            minv, maxv = min(beginidx, endidx), max(beginidx, endidx)
            self.h5annotation[h5key][:] = np.array([1, minv, maxv])
            
        self.h5annotation.flush()
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
            self.VideoThread.speed = max(self.VideoThread.speed-0.1, 0.1)

    @Slot()
    def UpdateSampleInfo(self, Count, m_list):
        if Count > 0:
            self.Count = Count
            
        self.ui.m_len_combo.clear()
        if len(m_list) > 0:
            items = [str(x) for x in m_list]
            self.ui.m_len_combo.addItems(items)
            self.ui.m_len_combo.setCurrentIndex(0)
    
    @Slot()
    def ChooseRecordFile(self):
        results = QFileDialog.getOpenFileName(self,
                                              "Open recordfile", 
                                              "../data/spbsl",
                                              "record Files (*.rec)")
        
        filepath = results[0]
        if os.path.exists(filepath):
            self.VideoThread.ReadRecordFile(filepath)

    @Slot()
    def closeEvent(self, event):
        magbox = QMessageBox.question
        reply = magbox(self, '警告', '系统将退出，是否确认', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.VideoThread.terminate()
            event.accept()
            self.h5annotation.close()
        else:
            event.ignore()


if __name__ == "__main__":
    import platform

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', action='store_true')

    args = parser.parse_args()
    server = args.server

    if server is True:
        videodir = '/home/mario/signdata/spbsl/normal'
    else:
        if platform.system().lower() == 'windows':
            videodir = 'F:/signdata/normal/video'
        else:
            videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'

    worddictpath = '../data/spbsl/WordDict.pkl'
    subdictpath = '../data/spbsl/SubtitleDict.pkl'
    recpoint = [(700, 100), (1280, 720)]
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())