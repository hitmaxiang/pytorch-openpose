'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-25 17:55:59
LastEditors: mario
LastEditTime: 2021-01-11 16:26:30
'''

import os
import cv2
import joblib
import utilmx
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

from SubtitleDict import WordsDict
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


def GetRandomImages(videopath, motionfilepath, framenum, rec):
    basedir, filename = os.path.split(videopath)
    videokey = filename[:3]

    # get the motion data
    posekey = 'posedata/pose/%s' % videokey
    handkey = 'handdata/hand/%s' % videokey
    motionfile = h5py.File(motionfilepath, 'r')
    posedata = motionfile[posekey]
    handdata = motionfile[handkey]

    # open the videofile and random choose framenum imges
    video = cv2.VideoCapture(videopath)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indexes = np.random.randint(count, size=(framenum,))

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


def WordFreqStatistics(recordfile, wordlistfile, wantnum):
    # recordinfodict = utilmx.ShapeletRecords().ReadRecordInfo(recordfile)
    with open(wordlistfile, 'r') as f:
        wordlines = f.readlines()
    
    counter = []
    for line in wordlines:
        if line.strip() == '':
            continue

        word, count = line.strip().split()
        counter.append(int(count))
    
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    
    ax.hist(counter, bins=100, range=(20, 1000), density=True)
    ax.set_xlabel('occurrence number')
    ax.set_ylabel('frequency density')
    fig.tight_layout()
    plt.savefig('../data/img/wordfreq.jpg')


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


def CalculateRecallRate(annotationh5file, recordfile, best_k=1):
    recorddict = utilmx.ShapeletRecords().ReadRecordInfo(recordfile)
    annotationfile = h5py.File(annotationh5file, 'r')

    RecallDict = {}
    RecallDict['global'] = []
    for word in annotationfile.keys():
        if word not in recorddict.keys():
            continue
        for indexkey in annotationfile[word].keys():
            annotation = annotationfile[word][indexkey][:]
            if annotation[0] == -1:  # negative
                continue
            real_begin, real_end = annotation[1:3]
            
            RecallDict['global'].append(0)
            for m_len in recorddict[word].keys():
                if m_len not in RecallDict.keys():
                    RecallDict[m_len] = []

                RecallDict[m_len].append(0)

                locs = recorddict[word][m_len]['loc']
                # 针对的是每个Word每个m长度下的最优的 k 个结果
                for loc in locs[:min(best_k, len(locs))]:
                    begin = loc[int(indexkey)]
                    end = begin + int(m_len)
                    if not(end < real_begin or begin > real_end):
                        RecallDict['global'][-1] = 1
                        RecallDict[m_len][-1] = 1
                        break
    
    for key in RecallDict.keys():
        correct = sum(RecallDict[key])
        Number = len(RecallDict[key])
        rate = correct/Number
        print('the recall rate of %s is %d/%d = %f' % (key, correct, Number, rate))


def locatIndexByVideoKeyoffset(h5pyfile, worddictpath, subdictpath, outpath):
    worddict = WordsDict(worddictpath, subdictpath)
    annotionfile = h5py.File(h5pyfile, 'r')

    newfile = h5py.File(outpath, 'w')
    for word in annotionfile.keys():
        sampleinfos = worddict.ChooseSamples(word, 1.5)
        for videokey in annotionfile[word].keys():
            for offset in annotionfile[word][videokey].keys():
                data = annotionfile[word][videokey][offset][:]
                for index, info in enumerate(sampleinfos):
                    if info[-1] == 1:
                        if info[0] == videokey and info[1] == int(offset):
                            newkey = '%s/%d' % (word, index)
                            newfile.create_dataset(newkey, data=data)
                            break
    
    annotionfile.close()
    newfile.close()


def AnticolorOfPicture(imgpath, outpath=None, mode=0):
    if os.path.isfile(imgpath):
        if mode == 0:
            # 将图片的黑白色进行翻转
            black = np.array([0, 0, 0])
            white = np.array([255, 255, 255])
            sigma = 5

            img = cv2.imread(imgpath)
            H, W, C = img.shape
            for h in range(H):
                for w in range(W):
                    color = img[h, w, :3]
                    if np.max(np.abs(color - black)) < sigma:
                        img[h, w, :3] = white
                    elif np.max(np.abs(color - white)) < sigma:
                        img[h, w, :3] = black
            cv2.imshow('img', img)
            cv2.waitKey(0)
        
        elif mode == 1:
            # 将所有颜色进行反色处理
            img = cv2.imread(imgpath)
            H, W, C = img.shape
            for h in range(H):
                for w in range(W):
                    color = img[h, w, :3]
                    img[h, w, :3] = 255 - color
            cv2.imshow('img', img)
            cv2.waitKey(0)
        
        if outpath is not None:
            cv2.imwrite(outpath, img)


def RunTest(testcode, server):
    if server:
        videodir = '/home/mario/signdata/spbsl/normal'
    else:
        videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'
    
    # extracted file from videos
    motion_dictpath = '../data/spbsl/motiondata.hdf5'
    worddictpath = '../data/spbsl/WordDict.pkl'
    subtitledictpath = '../data/spbsl/SubtitleDict.pkl'
    Recpoint = [(700, 100), (1280, 720)]

    # the data of shapelet info
    shapeletrecordED = '../data/spbsl/shapeletED.rec'
    shapeletrecordNet = '../data/spbsl/shapeletNetED.rec'

    # groud truth of the database
    annotationpath = '../gui/annotation.hdf5'
    wordlistpath = '../gui/wordlist.txt'
    wordinfopth = '../data/spbsl/wordinfo.txt'
    
    # target directory
    imgfolder = '../data/img'
    
    # save the 3D thumbnails of the video and skeleton
    if testcode == 0:
        getnum, framenum, shift = 6, 5, 10
        filenames = os.listdir(videodir)
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videodir, filename)

                for i in range(getnum):
                    outname = 'videothumb_%d_%d.jpg' % (framenum, i)
                    outpath = os.path.join(imgfolder, outname)
                    orisamples, samples = GetRandomImages(videopath, motion_dictpath, framenum, Recpoint)
                    ConstructImageStack(orisamples, outpath, shift)

                    comname = 'combthumb_%d_%d.jpg' % (framenum, i)
                    comname = os.path.join(imgfolder, comname)
                    ConstructImageStack(samples, comname, shift)
                break  # only onetime
    
    # draw the distribution of the word frequence
    elif testcode == 1:
        recordfile = '../data/spbsl/shapeletED.rec'
        wordinfo = '../data/spbsl/wordinfo.txt'
        WordFreqStatistics(recordfile, wordinfo, 1000)
    
    # caculate the recall rate of the shapelets
    elif testcode == 2:
        newannotationfile = '../data/spbsl/annotationindex.hdf5'
        if not os.path.exists(newannotationfile):
            locatIndexByVideoKeyoffset(annotationpath, worddictpath, subtitledictpath, newannotationfile)
        CalculateRecallRate(newannotationfile, shapeletrecordED)
        CalculateRecallRate(newannotationfile, shapeletrecordNet)
    
    elif testcode == 3:
        imgpath = '../data/img/keypoints_pose_18.png'
        outpath = '../data/img/keypoints_pose.png'
        AnticolorOfPicture(imgpath, outpath, mode=1)

        
if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-t', '--testcode', type=int, default=3)
    Parser.add_argument('-s', '--server', action='store_true')

    args = Parser.parse_args()
    testcode = args.testcode
    server = args.server
    
    RunTest(testcode, server)