'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-25 17:55:59
LastEditors: mario
LastEditTime: 2021-01-04 20:57:06
'''

import os
import cv2
import utilmx
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt 

from copy import deepcopy


def ConstructImageStack(images, outname, shift=10):

    N, H, W = images.shape[:3]
    x = np.arange(0, W)
    z = np.arange(0, H)

    X, Z = np.meshgrid(x, z)

    plt.switch_backend('agg')
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')

    Y = np.zeros_like(X)

    for i in range(N):
        # 归一化转化为【0-1】颜色
        colormat = np.reshape(images[i], (-1, 3))/255.0
        stride = shift * i
        ax.scatter(X + stride, Y + stride, Z + stride, c=colormat, s=0.5)

    ax.grid(None)
    # print('elev %f' % ax.elev)
    # print('azim %f' % ax.azim)
    # elev 表示 z 的 rotation，默认是向下低头 30（30）
    # zim 表示 xy 面的 rotation， 默然是向 左旋转 30 （即-60），
    ax.view_init(elev=15, azim=-75)
    plt.axis('off')
    plt.savefig(outname)
    print('the fig %s is saved!' % outname)
    # plt.show()


def GetRandomImages_from_video(videopath, frames, rec=None, fx=1):
    if os.path.exists(videopath):
        video = cv2.VideoCapture(videopath)
        Count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        indexes = np.random.randint(Count, size=(frames,))
        images = []
        for i in range(frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, indexes[i])
            _, img = video.read()
            img = cv2.resize(img, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            if rec is not None:
                img = img[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0], :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转化为 RGB 颜色空间
                img = cv2.flip(img, 0)  # 垂直翻转
            images.append(img)
        images = np.array(images)
        video.release()
    
        return images


def GetrandomImages_from_video_with_skeleton(videopath, posedata, handdata, framnum, rec):
    if os.path.exists(videopath):
        video = cv2.VideoCapture(videopath)
        Count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        indexes = np.random.randint(Count, size=(framnum,))

        images = []
        origimgs = []
        
        for index in indexes:
            video.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, img = video.read()
            img = img[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0], :]

            origimg = deepcopy(img)
            origimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2RGB)  # 转化为 RGB 颜色空间
            origimg = cv2.flip(origimg, 0)  # 垂直翻转
            origimgs.append(deepcopy(origimg))

            img = utilmx.DrawPose(img, posedata[index], handdata[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转化为 RGB 颜色空间
            img = cv2.flip(img, 0)  # 垂直翻转
            images.append(img)
        
        return np.array(origimgs), np.array(images)


def GetRandomImages_of_skeleton(motion_dictpath, framnum, shape):
    h5file = h5py.File(motion_dictpath, mode='r')
    while True:
        videokey = '%03d' % (np.random.randint(100))
        posekey = 'posedata/pose/%s' % videokey
        handkey = 'handdata/hand/%s' % videokey

        if posekey in h5file and handkey in h5file:
            posedata = h5file[posekey]
            handdata = h5file[handkey]
            break
    
    Count = len(posedata)
    indexes = np.random.randint(Count, size=(framnum,))
    
    images = []
    for i in indexes:
        img = np.zeros(shape + (3,), dtype=np.uint8)
        # set the background (BGR order)
        img[:, :, 0] = 105
        img[:, :, 1] = 118
        img[:, :, 2] = 128
        img = utilmx.DrawPose(img, posedata[i], handdata[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转化为 RGB 颜色空间
        img = cv2.flip(img, 0)  # 垂直翻转
        images.append(img)
    
    return np.array(images)


def WordFreqStatistics(recordfile, wordlistfile, wantnum):
    recordinfodict = utilmx.ShapeletRecords().ReadRecordInfo(recordfile)
    with open(wordlistfile, 'r') as f:
        wordlines = f.readlines()
    
    WordStatisticList = []
    for line in wordlines:
        if line.strip() == '':
            continue

        word, count = line.strip().split()
        if word not in recordinfodict.keys():
            continue

        bestsocre = 0
        for key in recordinfodict[word].keys():
            for shapelet in recordinfodict[word][key]['shapelet']:
                bestsocre = max(bestsocre, shapelet[-1])
        
        WordStatisticList.append([word, count, bestsocre])
    
    WordStatisticList.sort(key=lambda item: item[-1], reverse=True)
    WordStatisticList = WordStatisticList[:wantnum]
    counter = [x[1] for x in WordStatisticList]
    
    plt.switch_backend('agg')
    plt.hist(counter, bins=20)
    plt.savefig('../data/img/wordinfofreq.jpg')


def ChooseAnnotatedWordFromRecord(recordfile, outname, thre=20, number=1000):
    recordinfodict = utilmx.ShapeletRecords().ReadRecordInfo(recordfile)
    wordlist = []
    
    for word in recordinfodict.keys():
        bestscore = 0
        for m_len in recordinfodict[word].keys():
            for shapelet in recordinfodict[word][m_len]['shapelet']:
                bestscore = max(bestscore, shapelet[-1])
        if len(recordinfodict[word][m_len]['loc'][0]) >= thre * 2:
            wordlist.append([word, bestscore])

    wordlist.sort(key=lambda item: item[-1], reverse=True)

    chooseword = [x[0] for x in wordlist[:min(len(wordlist), number)]]
    
    with open(outname, 'w') as f:
        for word in chooseword:
            f.write('%s\n' % word)


def RunTest(testcode, server):
    if server:
        videodir = '/home/mario/signdata/spbsl/normal'
    else:
        videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'

    motion_dictpath = '../data/spbsl/motiondata.hdf5'
    imgfolder = '../data/img'
    Recpoint = [(700, 100), (1280, 720)]
    
    if testcode == 0:
        # get the 3D thumbnails of the video frames
        getnum, framenum = 5, 5
        filenames = os.listdir(videodir)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videodir, filename)
                for i in range(getnum):
                    outname = 'videothumb_%d_%d.jpg' % (framenum, i)
                    outpath = os.path.join(imgfolder, outname)
                    samples = GetRandomImages_from_video(videopath, framenum, Recpoint)
                    ConstructImageStack(samples, outpath)
                break  # only onetime
    
    elif testcode == 1:
        # save the 3D thumbnails of the skeleton frames
        getnum, framenum = 10, 5
        imgshape = (Recpoint[1][1] - Recpoint[0][1], Recpoint[1][0] - Recpoint[0][0])
        for i in range(getnum):
            outname = 'skeletonthumb_%d_%d.jpg' % (framenum, i)
            outpath = os.path.join(imgfolder, outname)
            samples = GetRandomImages_of_skeleton(motion_dictpath, framenum, imgshape)
            ConstructImageStack(samples, outpath)
    
    elif testcode == 2:
        # save the 3D thumbnails of the video and skeleton
        getnum, framenum, shift = 6, 5, 10
        filenames = os.listdir(videodir)

        motionh5file = h5py.File(motion_dictpath, 'r')

        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videodir, filename)

                videokey = filename[:3]
                posekey = 'posedata/pose/%s' % videokey
                handkey = 'handdata/hand/%s' % videokey

                posedata = motionh5file[posekey]
                handdata = motionh5file[handkey]

                for i in range(getnum):
                    outname = 'videothumb_%d_%d.jpg' % (framenum, i)
                    outpath = os.path.join(imgfolder, outname)
                    orisamples, samples = GetrandomImages_from_video_with_skeleton(
                        videopath, posedata=posedata, handdata=handdata, framnum=framenum, rec=Recpoint)
                    
                    ConstructImageStack(orisamples, outpath, shift)

                    comname = 'combthumb_%d_%d.jpg' % (framenum, i)
                    comname = os.path.join(imgfolder, comname)
                    ConstructImageStack(samples, comname, shift)
                break  # only onetime
    
    elif testcode == 3:
        recordfile = '../data/spbsl/shapeletED.rec'
        wordinfo = '../data/spbsl/wordinfo.txt'
        WordFreqStatistics(recordfile, wordinfo, 1000)
    
    elif testcode == 4:
        recordfile = '../data/spbsl/shapeletED.rec'
        wordlist = '../gui/wordlist.txt'

        ChooseAnnotatedWordFromRecord(recordfile, wordlist)
        
    
if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-t', '--testcode', type=int, default=4)
    Parser.add_argument('-s', '--server', action='store_true')

    args = Parser.parse_args()
    testcode = args.testcode
    server = args.server
    
    RunTest(testcode, server)