'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-25 17:55:59
LastEditors: mario
LastEditTime: 2021-01-11 16:26:30
'''

import os
import re
import cv2
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


def CalculateRecallRate_h5file(annotationh5file, h5recordfile, threhold=0, sigma=0):
    recordfile = h5py.File(h5recordfile, 'r')
    annotationfile = h5py.File(annotationh5file, 'r')

    AllDict = {}
    AllDict['global'] = []
    AllDict['major'] = {}
    AllDict['major']['global'] = []
    AllDict['minor'] = {}
    AllDict['minor']['global'] = []

    # 为确定每个positive 样本是否被找到
    for word in annotationfile.keys():
        if word not in recordfile.keys():
            continue
        idxkey = '%s/sampleidxs' % word
        vdokey = '%s/videokeys' % word
        rangeindexs = recordfile[idxkey][:]
        videoindexs = recordfile[vdokey][:]

        # get the corresponding relationship between record and annotation
        Idxs = GetIdxofAnnotation(annotationfile[word], rangeindexs, videoindexs)
        Posidxs = [x for x in Idxs if x[-2] == 1]
        ratio = len(Posidxs)/len(Idxs)

        if ratio >= threhold:
            group = 'major'
        else:
            group = 'minor'

        # 计算positive sequences 被找到的概率
        for item in Posidxs:
            videokey, offset, label, idx = item
            annodata = annotationfile[word][videokey][offset][:]
            real_begin, real_end = annodata[1:3]

            AllDict['global'].append(0)
            AllDict[group]['global'].append(0)

            for m_len in recordfile[word].keys():
                if re.match(r'^\d+$', m_len) is None:
                    continue
                
                if m_len not in AllDict[group].keys():
                    AllDict[group][m_len] = []
                AllDict[group][m_len].append(0)
                
                locs = recordfile[word][m_len]['locs']
                begin = locs[int(idx)]
                end = begin + int(m_len)
                if JudgeWhetherFound((real_begin, real_end), (begin, end), sigma) == 1:
                    AllDict['global'][-1] = 1
                    AllDict[group]['global'][-1] = 1
                    AllDict[group][m_len][-1] = 1

    ratelist = AllDict['global']
    print('the total global rate is %f' % (sum(ratelist)/len(ratelist)))
    # list the information of majority words
    for group in ['major', 'minor']:
        for key in AllDict[group].keys():
            ratelist = AllDict[group][key]
            if len(ratelist) == 0:
                continue
            print('the %s group global rate of %s is %f' % (group, key, sum(ratelist)/len(ratelist)))


def DistAnalysis(h5recordfile, h5annotationfile):
    h5recfile = h5py.File(h5recordfile, mode='r')
    h5annfile = h5py.File(h5annotationfile, mode='r')

    m_len_pattern = r'^\d+$'

    words = h5recfile.keys()
    for word in words:
        if word not in h5recfile.keys():
            continue

        idxkey = '%s/sampleidxs' % word
        vdokey = '%s/videokeys' % word
        rangeindexs = h5recfile[idxkey][:]
        videoindexs = h5recfile[vdokey][:]

        # 获取该Word的标签在record文件中的索引位置
        IDXes = GetIdxofAnnotation(h5annfile[word], rangeindexs, videoindexs)
        posidx, annorange = [], []
        for IDX in IDXes:
            videokey, offset, label, idx = IDX
            if idx != -1:
                if label == 1:
                    posidx.append(idx)
                    annorange.append(h5annfile['%s/%s/%s' % (word, videokey, offset)][1:])

        for mlenkey in h5recfile[word].keys():
            if re.match(m_len_pattern, mlenkey) is not None:
                dists = h5recfile['%s/%s/dists' % (word, mlenkey)][:]
                locs = h5recfile['%s/%s/locs' % (word, mlenkey)][:]

                for i, idx in enumerate(posidx):
                    matched = JudgeWhetherFound(annorange[i], [locs[idx], locs[idx] + int(mlenkey)])
                    plt.scatter([dists[idx]], [1], c='r', marker='*')
                    if matched == 1:
                        plt.scatter([dists[idx]], [1.5], c='y', marker='o')
                
                for idx in range(len(dists)):
                    plt.scatter([dists[idx]], [2], c='b', marker='*')
            
                plt.title('%s:%s' % (word, mlenkey))
                plt.show()
        

def GetIdxofAnnotation(annogroup, rangelist, videolist):
    # IDX 的格式为: videokey, offset, label, index(该sample在整个中的位置)
    IDX = []
    for videokey in annogroup.keys():
        for offset in annogroup[videokey].keys():
            data = annogroup[videokey][offset]
            IDX.append([videokey, offset, -1, -1])  # 后两个是label, 以及在list中的位置
            for idx in range(len(videolist)):
                if str(videolist[idx], encoding='utf8') == videokey and rangelist[idx][0] == int(offset):
                    if data[0] == 1:
                        IDX[-1][2] = 1
                    # 确定该 annotation 在记录中的位置
                    IDX[-1][-1] = idx
                    break
    return IDX


def JudgeWhetherFound(annorange, locrange, sigma=0):
    # 判断两个区域范围的重叠区域是否大于一个比例阈值: sigma
    a, b = annorange
    x, y = locrange
    if y < a or x > b:  # 这种情况下, 两者不重叠
        return -1
    elif x >= a and y <= b:  # 这种情况下, locrange 完全在 annorange 的内部
        return 1
    elif x < a:
        if (y - a) > sigma * (y - x):
            return 1
        else:
            return -1
    elif y > b:
        if (b - x) > sigma * (y - x):
            return 1
        else:
            return -1
    else:
        # 理论上应该不会有上面例外的情况了
        return None


# 这里主要是针对recordfile之前int16下溢出的record 进行修正
def RecordReindex(recordhdf5file, worddictpath, subtitlefilepath, outfilepath):
    fxpattern = r'fx:(\d+.\d+)-'
    worddict = WordsDict(worddictpath, subtitlefilepath)
    # copy the recordfilepath to outfilepath
    with open(recordhdf5file, 'rb') as rstream:
        content = rstream.read()
        with open(outfilepath, 'wb') as wstream:
            wstream.write(content)
    
    # 然后修改复制得到的文件
    recordfile = h5py.File(recordhdf5file)
    for word in recordfile.keys():
        # 获取当前文件的信息
        infokey = '%s/loginfo' % word
        idxkey = '%s/sampleidxs' % word
        infos = str(recordfile[infokey][:][0], encoding='utf8')
        fx = float(re.findall(fxpattern, infos)[0])
        indexes = recordfile[idxkey][:]
        
        # according the above inforamtion to get the true info
        num = len(indexes)
        sample_indexes = worddict.ChooseSamples(word, fx, maxitems=num)
        clipidx = np.array([x[1:3] for x in sample_indexes if x[-1] == 1]).astype(np.int32)
        utilmx.WriteRecords2File(outfilepath, idxkey, clipidx, (num, 2), dtype=np.int32)
        
    recordfile.close()


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

    h5shapeletrecordED = '../data/spbsl/shapeletED.hdf5'
    h5shapeletrecordNet = '../data/spbsl/shapeletNetED.hdf5'

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
    
    elif testcode == 3:
        imgpath = '../data/img/keypoints_pose_18.png'
        outpath = '../data/img/keypoints_pose.png'
        AnticolorOfPicture(imgpath, outpath, mode=1)
    
    elif testcode == 4:
        # calculate the recall rate with the all hdf5 file
        newannotationfile = '../data/spbsl/annotationindex.hdf5'
        # h5shapeletrecordED = '../data/spbsl/temprecord.hdf5'
        h5shapeletrecordED = '../data/spbsl/bk_shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk_shapeletNetED.hdf5'
        if not os.path.exists(newannotationfile):
            locatIndexByVideoKeyoffset(annotationpath, worddictpath, subtitledictpath, newannotationfile)
        CalculateRecallRate_h5file(annotationpath, h5shapeletrecordED, 0.4)
        CalculateRecallRate_h5file(annotationpath, h5shapeletrecordNet, 0.4)
    
    elif testcode == 5:
        h5shapeletrecordED = '../data/spbsl/bk_shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk_shapeletNetED.hdf5'
        outh5shapeletrecordED = '../data/spbsl/bk1_shapeletED.hdf5'
        # outh5shapeletrecordNet = '../data/spbsl/bk1_shapeletNetED.hdf5'

        # RecordReindex(h5shapeletrecordED, worddictpath, subtitledictpath, outh5shapeletrecordED)
        CalculateRecallRate_h5file(annotationpath, outh5shapeletrecordED, threhold=0.5, sigma=0.5)
    
    elif testcode == 6:
        h5shapeletrecordED = '../data/spbsl/bk1_shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk3_shapeletNetED.hdf5'
        # DistAnalysis(h5shapeletrecordED, annotationpath)
        DistAnalysis(h5shapeletrecordNet, annotationpath)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-t', '--testcode', type=int, default=5)
    Parser.add_argument('-s', '--server', action='store_true')

    args = Parser.parse_args()
    testcode = args.testcode
    server = args.server
    
    RunTest(testcode, server)