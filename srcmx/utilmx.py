'''
Description: the common used utilization functions
Version: 2.0
Autor: mario
Date: 2020-08-27 20:41:43
LastEditors: mario
LastEditTime: 2020-09-25 14:32:21
'''
import os
import re
import cv2
import numpy as np
from tslearn import metrics
from scipy.io import loadmat
from numba import jit
from sklearn.preprocessing import scale


def FaceDetect(frame):
    '''
    description: detect the face with the opencv trained model
    param: 
        frame: the BGR picture
    return: the facecenter and the face-marked frame
    author: mario
    '''
    face_cascade = cv2.CascadeClassifier()
    profileface_cascade = cv2.CascadeClassifier()

    # load the cascades
    face_cascade.load(cv2.samples.findFile('./model/haarcascade_frontalface_alt.xml'))
    profileface_cascade.load(cv2.samples.findFile('./model/haarcascade_profileface.xml'))

    # convert the frame into grayhist
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) == 0:
        faces = profileface_cascade.detectMultiScale(frame_gray)
    
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 0], 2)
    return faces, frame


def Getvideoframes(dirname):
    '''
    description: get the infomation of the video framnumbers, write the file with the format
        ['videoname, framcounts, maximumframeinsubtitle']
    param: dirname, the filepath of the videofile
    return {type} 
    author: mario
    '''
    # extract the startframe information
    starfile = '../data/bbcposestartinfo.txt'
    startdict = {}
    # define the re pattern of the video index and startframe
    pattern1 = r'of\s*(\d+)\s*'
    pattern2 = r'is\s*(\d+).*$'
    with open(starfile) as f:
        lines = f.readlines()
        for line in lines:
            videoindex = int(re.findall(pattern1, line)[0])
            startindex = int(re.findall(pattern2, line)[0])
            startdict[videoindex] = startindex

    # extract the framecount info
    countdict = {}
    for i in range(1, 93):
        videopath = os.path.join(dirname, 'e%d.avi' % i)
        video = cv2.VideoCapture(videopath)
        if video.isOpened():
            count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            countdict[i] = count
        video.release()
    
    # get the maximaum frameindex from the dictinfo
    maxframdict = {}
    matfile = '../data/bbc_subtitles.mat'
    mat = loadmat(matfile)
    datas = mat['bbc_subtitles']

    for index, data in enumerate(datas[0]):
        if len(data[1][0]) == 0:
            maxframdict[index+1] = 0
        else:
            lastrecord = data[1][0][-1]
            maxframdict[index+1] = lastrecord[1][0][0]

    with open('../data/videoframeinfo.txt', 'w') as f:
        for i in list(countdict.keys()):
            f.write('e%d.avi\t%6d\t%6d\t%6d\n' % (i, startdict[i], countdict[i], maxframdict[i]))


def ReadRecord(recordfile):
    '''
    description: extract the data (test data) from the record data
    param {type} 
    return {type} 
    author: mario
    '''
    RecordsDict = {}
    with open(recordfile, 'r') as f:
        lines = f.readlines()
    params_pattern = r'lenth:\s*(\d+),\s*iters:\s*(\d+),\s*feature:\s*(\d+)'
    score_pattern = r'score:\s*(\d*.\d+)$'
    locs_pattern = r'Locs:(.*)$'
    for line in lines:
        params = re.findall(params_pattern, line)
        if len(params) != 0:
            key = '-'.join(params[0])
            if key not in RecordsDict.keys():
                RecordsDict[key] = {'num': 1}
                RecordsDict[key]['score'] = []
                RecordsDict[key]['location'] = []
            else:
                RecordsDict[key]['num'] += 1
            continue

        score = re.findall(score_pattern, line)
        if len(score) != 0:
            RecordsDict[key]['score'].append(float(score[0]))
            continue
        
        locs = re.findall(locs_pattern, line)
        if len(locs) != 0:
            locs = locs[0].split()
            locs = [int(d) for d in locs]
            RecordsDict[key]['location'].append(locs)
        
    return RecordsDict


if __name__ == "__main__":
    ReadRecord('../data/record.txt')