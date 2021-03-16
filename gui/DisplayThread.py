'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-23 15:06:20
LastEditors: mario
LastEditTime: 2021-03-16 15:01:40
'''
import sys
sys.path.append('..')

import os
import cv2
import numpy as np
import time
from PySide2.QtCore import QThread, Signal, Slot
from srcmx import SubtitleDict
from srcmx import utilmx
from threading import Thread


class DisplayThread(QThread):
    # word, videonum, offset, length, index
    StatesOutSignal = Signal(str, str, int, int, int)
    # currentindex, speed, shapelet_begin
    ImgDisplaySignal = Signal(int, float, int)

    # positive_Number, m_len_list
    SampleInfoSignal = Signal(int, list)

    def __init__(self, worddictpath, subdictpath, videodir, recpoint):
        super().__init__()
        # load the data
        self.worddict = SubtitleDict.WordsDict(worddictpath, subdictpath)
        self.videohandledict = self.Getvideohandles(videodir)
        
        self.recpoint = recpoint
        # self.gui = gui

        # control variables
        self.wordlist = []
        self.word = None
        self.ShapeletDict = {}
        self.random = True
        
        self.speed = 1.0
        self.duration = 0.033
        self.working = True
        self.wordloop = True
        self.sampleloop = True
        self.displayloop = True

        self.maximumindex = 0
        self.minimumindex = 0
        self.currentindex = 0

        self.m_len = 0

        # share img
        h, w = recpoint[1][1] - recpoint[0][1], recpoint[1][0] - recpoint[0][0]
        self.shareImg = np.zeros((h, w, 3), dtype=np.uint8)
        self.ShapeletImg = np.zeros_like(self.shareImg)
    
    def Getvideohandles(self, videodir):
        videohandles = {}
        videofiles = os.listdir(videodir)
        for videofile in videofiles:
            if videofile.endswith('mp4'):
                filepath = os.path.join(videodir, videofile)
                videohandles[videofile[:3]] = cv2.VideoCapture(filepath)
        return videohandles

    @Slot()
    def ReadRecordFile(self, recodfile):
        self.ShapeletDict = utilmx.ShapeletRecords().ReadRecordInfo(recodfile)
        m_list = []
        if self.word in self.ShapeletDict.keys():
            m_list = list(self.ShapeletDict[self.word].keys())
            if len(m_list) >= 1:
                self.m_len = m_list[0]
                self.m_list = m_list
                self.SampleInfoSignal.emit(-1, m_list)

    @Slot()
    def UpdateWord(self, word, mode):
        self.wordlist = [word]
        self.wordloop = False

    @Slot()
    def UpdateLoopRange(self, minindex, maxindex):
        # print('get the signal of %d--%d' % (minindex, maxindex))
        self.minimumindex = minindex
        self.maximumindex = maxindex
        self.currentindex = self.minimumindex
    
    @Slot()
    def NextSample(self):
        self.displayloop = False
    
    @Slot()
    def NextWord(self):
        self.sampleloop = False

    @Slot()
    def SetCurrent_m_len(self, m_len):
        self.m_len = m_len

    def run(self):
        recpoint = self.recpoint
        h, w = recpoint[1][1] - recpoint[0][1], recpoint[1][0] - recpoint[0][0]

        while self.working:
            # self.wait(2000)
            # print("i'm in the loop")
            time.sleep(1)
            self.wordloop = True
            for word in self.wordlist:
                self.word = word
                if not self.wordloop:
                    break
                
                samples = self.worddict.ChooseSamples(word, 1.5)
                # only use the positive samples
                samples = [x for x in samples if x[-1] == 1]

                # 提取 shapelet record 文件中的 word 信息
                m_list = []
                if word in self.ShapeletDict.keys():
                    m_list = list(self.ShapeletDict[word].keys())
                    if len(m_list) >= 1:
                        self.m_len = m_list[0]
                self.SampleInfoSignal.emit(len(samples), m_list)
                self.m_list = m_list

                self.sampleloop = True

                # 遍历该 Word 的所有 samples
                for indx, sample in enumerate(samples):
                    if not (self.wordloop and self.sampleloop):
                        break

                    keynum, begin, end, label = sample
                    length = end - begin
                    videoclips = np.zeros((length, h, w, 3), dtype=np.uint8)
                    self.videohandledict[keynum].set(cv2.CAP_PROP_POS_FRAMES, begin)
                    for i in range(length):
                        _, frame = self.videohandledict[keynum].read()
                        frame = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]
                        videoclips[i] = frame
                    
                    # word, videonum, offset, length, index
                    self.StatesOutSignal.emit(word, keynum, begin, length, indx)
                    self.currentindex = 0
                    self.displayloop = True

                    # display loop
                    while self.displayloop and self.sampleloop and self.wordloop:
                        if self.currentindex >= self.maximumindex or self.currentindex < self.minimumindex:
                            self.currentindex = self.minimumindex
                            time.sleep(0.5)
                        
                        # set the loop of shapelet
                        if self.m_len in self.m_list:
                            shapelet_begin = self.ShapeletDict[word][self.m_len]['loc'][0][indx]
                            shapelet_end = int(self.m_len) + shapelet_begin
                        else:
                            shapelet_begin, shapelet_end = 0, 0
                            shapelt_current = 0

                        # display the shapelet
                        if shapelt_current >= shapelet_end or shapelt_current < shapelet_begin:
                            shapelt_current = shapelet_begin
                        
                        self.ShapeletImg = videoclips[shapelt_current]
                        shapelt_current += 1

                        # print(self.currentindex, self.maximumindex)
                        self.shareImg = videoclips[self.currentindex]
                        self.ImgDisplaySignal.emit(self.currentindex, self.speed, shapelet_begin)
                            
                        time.sleep(self.duration/self.speed)
                        self.currentindex += 1
                            