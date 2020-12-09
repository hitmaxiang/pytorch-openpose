'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-07 10:53:28
LastEditors: mario
LastEditTime: 2020-12-07 20:12:37
'''
import os
import re
import cv2
import time
import joblib
import argparse
import numpy as np
from SubtitleDict import WordsDict


# 从 shapelet 的解析记录中提取解析结果
def ReadShapeletReccords(filepath):
    headerpattern = r'header:.+word:(\w+)\s.+:\s*(\d+)$'
    shapeletpattern = r'(\d+)-.+index:(\d+).+offset:(\d+)-.+(\d\.\d+)$'
    recordpattern = r'^(\s*\d+\s*)+$'
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    RECORDDIC = {}
    word = None
    
    for line in lines:
        # search the header
        results = re.findall(headerpattern, line)
        if len(results) > 0:
            word, m_len = results[0]

            if word not in RECORDDIC.keys():
                RECORDDIC[word] = {}
            if m_len not in RECORDDIC[word].keys():
                RECORDDIC[word][m_len] = []
            continue
        
        # search the shapelet 
        results = re.findall(shapeletpattern, line)
        if len(results) > 0:
            videokey, begindex, offset, accuracy = results[0]
            continue

        # search the record results
        results = re.findall(recordpattern, line)
        if len(results) > 0:
            records = re.split('\t', line.strip())
            
            # store the extrated results
            shapelet = {}
            shapelet['videokey'] = videokey
            shapelet['beginindex'] = int(begindex) + int(offset)
            shapelet['m_len'] = int(m_len)
            shapelet['accuracy'] = float(accuracy)
            shapelet['records'] = [int(x) for x in records]
            
            RECORDDIC[word][m_len].append(shapelet)

    return RECORDDIC

    
# 校验提取出来的 shapelet 的有效性
def SignInstancesDemons(worddictpath, subdictpath, resultspath, videodir, recpoint):
    worddict = WordsDict(worddictpath, subdictpath)
    resultdict = ReadShapeletReccords(resultspath)
    h, w = recpoint[1][1] - recpoint[0][1], recpoint[1][0] - recpoint[0][0]
    
    keywords = list(resultdict.keys())

    # open all the videos and store the handle
    videohandles = {}
    videofiles = os.listdir(videodir)
    for videofile in videofiles:
        if videofile.endswith('mp4'):
            filepath = os.path.join(videodir, videofile)
            videohandles[videofile[:3]] = cv2.VideoCapture(filepath)

    while True:
        word = np.random.choice(keywords)
        samples = worddict.ChooseSamples(word, 1.5)
        labels = [x[-1] for x in samples]

        # 去寻找最好的 shapelet
        Bestshapelet = None
        for m_len in resultdict[word].keys():
            shapelet = resultdict[word][m_len][-1]
            if Bestshapelet is None or (shapelet['accuracy'] >= Bestshapelet['accuracy']):
                Bestshapelet = shapelet
    
        videokey = Bestshapelet['videokey']
        beginindex = Bestshapelet['beginindex']
        accuracy = Bestshapelet['accuracy']
        records = Bestshapelet['records']
        m_len = Bestshapelet['m_len']
        
        print('%s----%f' % (m_len, accuracy))

        VideoClips = np.zeros((sum(labels)+1, int(m_len), h, w, 3), dtype=np.uint8)

        tempsamples = [[videokey, beginindex, 0, 1]] + samples
        records.insert(0, 0)
        
        for index, sample in enumerate(tempsamples):
            keynum, begin, end, label = sample
            if label != 1:
                continue
            beginindex = begin + records[index]
            videohandles[keynum].set(cv2.CAP_PROP_POS_FRAMES, beginindex)
            for i in range(int(m_len)):
                ret, frame = videohandles[keynum].read()
                img = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]
                VideoClips[index, i] = img
        
        # 演示
        for i in range(1, len(VideoClips)):
            number = 0
            count = 0
            while True:
                number = number % int(m_len)
                cv2.imshow('shapelet', VideoClips[0, number])
                cv2.imshow('instance', VideoClips[i, number])
                q = cv2.waitKey(100) & 0xff
                if q == ord('q') or count > 10:
                    break
                number += 1
                if number == int(m_len):
                    count += 1                     


if __name__ == "__main__":
    # ReadShapeletReccords('../data/spbsl/shapeletED.rec')
    worddictpth = '../data/spbsl/WordDict.pkl'
    subtitlepth = '../data/spbsl/SubtitleDict.pkl'
    videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'
    resultspth = '../data/spbsl/shapeletED.rec'
    Recpoint = [(700, 100), (1280, 720)]

    SignInstancesDemons(worddictpth, subtitlepth, resultspth, videodir, Recpoint)
