'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-23 15:06:20
LastEditors: mario
LastEditTime: 2020-12-23 20:26:44
'''
import sys
sys.path.append('..')

import os
import cv2
import numpy as np
import time
from PySide2.QtCore import QThread, Signal, Slot
from srcmx import SubtitleDict
from threading import Thread


class DisplayThread(QThread):
# class DisplayThread(Thread):

    # word, videonum, offset, length, speed, demoindex
    StatesOutSignal = Signal(str, str, str, str, str, str)
    # PaintSignal = Signal(type(np.array([])))
    PaintSignal = Signal()
    
    def __init__(self, worddictpath, subdictpath, videodir, recpoint):
        super().__init__()
        # load the data
        self.worddict = SubtitleDict.WordsDict(worddictpath, subdictpath)
        self.videohandledict = self.Getvideohandles(videodir)
        
        self.recpoint = recpoint
        # self.gui = gui

        # control variables
        self.wordlist = ['supermarket']
        self.random = True
        self.number = 10
        self.displaying = False
        self.working = True
        self.speed = 1.0
        self.duration = 0.200
        self.loop = True

        # share img
        h, w = recpoint[1][1] - recpoint[0][1], recpoint[1][0] - recpoint[0][0]
        self.shareImg = np.zeros((h, w, 3), dtype=np.uint8)

    def Getvideohandles(self, videodir):
        videohandles = {}
        videofiles = os.listdir(videodir)
        for videofile in videofiles:
            if videofile.endswith('mp4'):
                filepath = os.path.join(videodir, videofile)
                videohandles[videofile[:3]] = cv2.VideoCapture(filepath)
        return videohandles

    @Slot()
    def UpdateWord(self, word, mode):
        self.wordlist = [word]

    def run(self):
        recpoint = self.recpoint
        h, w = recpoint[1][1] - recpoint[0][1], recpoint[1][0] - recpoint[0][0]

        while self.working:
            # self.wait(2000)
            print("i'm in the loop")
            time.sleep(1)
            for word in self.wordlist:
                # # 判断选定单词是否在其中
                # if word not in self.worddict.keys():
                #     continue

                samples = self.worddict.ChooseSamples(word, 1.5)
                counter = sum([x[-1] for x in samples])
                number = 0
                for sample in samples:
                    keynum, begin, end, label = sample

                    if label != 1:
                        continue
                    length = end - begin
                    videoclips = np.zeros((length, h, w, 3), dtype=np.uint8)
                    self.videohandledict[keynum].set(cv2.CAP_PROP_POS_FRAMES, begin)
                    for i in range(length):
                        _, frame = self.videohandledict[keynum].read()
                        frame = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]
                        videoclips[i] = frame
                    
                    # word, videonum, offset, length, speed, demoindex
                    self.StatesOutSignal.emit(word, keynum, str(begin), str(length),
                                              'x%f' % self.speed, '%d/%d' % (number, counter))
                    while self.loop:
                        print('i am in the display loop')
                        for i in range(length):
                            # self.gui.displaygui.videoplayer.DisplayImg(videoclips[i])
                            self.shareImg = videoclips[i]
                            self.PaintSignal.emit()
                            time.sleep(self.duration/self.speed)