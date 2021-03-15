'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-25 17:55:59
LastEditors: mario
LastEditTime: 2021-03-15 21:58:00
'''

import os
from platform import dist
import re
import cv2
from numpy import random
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


def CalRecallRateOneWord(annotationfilepath, shapeletfilepath, word, sigma, beta=1):
    # the structure of annotationfile is : word/videokey/offset: [label, begidx, endidx]
    pass
    annotationfile = h5py.File(annotationfilepath, mode='r')
    shapeletfile = h5py.File(shapeletfilepath, mode='r')

    # get the sample idx for the annotion record in the annotationfile
    annowordgroup = annotationfile[word]
    rangelist = shapeletfile[word]['sampleidxs']
    vidxlist = shapeletfile[word]['videokeys']
    # the structure of IDX is [videokey, offset, label, idx], idx 是该sample在shapeletrecord中的位置
    Idxs = GetIdxofAnnotation(annowordgroup, rangelist, vidxlist)

    # for a given word in the annotationfile, we should calculate its kinds of indexs

    # positive rate: the rate of the positive sample is true positive
    Posidxs = [x for x in Idxs if x[2] == 1]
    pos_ratio = len(Posidxs)/len(Idxs)

    # beta 是只分析距离最近的比例的pos samples
    pos_num = int(len(rangelist)*beta)

    # recall rate
    RecallDict = {}
    levelpattern = r'^\d+$'
    for levelkey in shapeletfile[word].keys():
        if re.match(levelpattern, utilmx.Encode(levelkey)) is not None:
            RecallDict[levelkey] = [0, 0]  # pos_num, foundnum
            locs = shapeletfile[word][levelkey]['locs'][:]

            dists = shapeletfile[word][levelkey]['dists'][:]
            sortindexs = list(np.argsort(dists))
            
            for item in Posidxs:
                videokey, offset, label, idx = item
                if sortindexs.index(idx) > pos_num:
                    continue
                RecallDict[levelkey][0] += 1

                a_begin, a_end = annotationfile[word][videokey][offset][1:]
                l_begin = locs[idx]
                l_end = l_begin + int(levelkey)
                
                if JudgeWhetherFound((a_begin, a_end), (l_begin, l_end), sigma) == 1:
                    RecallDict[levelkey][1] += 1

    annotationfile.close()
    shapeletfile.close()
    return pos_ratio, RecallDict


def ResultsAnalysis(annotationfilepath, shapeletfilepath, alpha=0.5, beta=0.3, sigma=0.5):
    annofile = h5py.File(annotationfilepath, mode='r')
    shapeletfile = h5py.File(shapeletfilepath, mode='r')

    PosTrue = [0, 0, 0]

    for word in annofile.keys():
        pos_ratio, recalldict = CalRecallRateOneWord(annotationfilepath, shapeletfilepath, word, sigma, beta)
        # just find the best score
        bestscore = 0.0
        level = ''
        for levelkey in recalldict.keys():
            if recalldict[levelkey][0] != 0:
                score = recalldict[levelkey][1]/recalldict[levelkey][0]
                if score > bestscore:
                    bestscore = score
                    level = levelkey
        # print("%s, pos_ratio:%f, bestscore %f with level %s" % (word, pos_ratio, bestscore, level))

        # using the score and avgdist to find the best score
        level = GetBestLevel(shapeletfile[word], beta)
        score = 0
        if recalldict[level][0] != 0:
            score = recalldict[level][1]/recalldict[level][0]
        # print("%s, pos_ratio:%f, score %f with level %s\n" % (word, pos_ratio, score, level))

        # bestscore = score 
        
        if pos_ratio > alpha:
            PosTrue[0] += 1
            if bestscore > 0.5:
                PosTrue[1] += 1
            if score > 0.5:
                PosTrue[2] += 1
    
    print('the best ture rate is %d/%d-->%f' % (PosTrue[1], PosTrue[0], PosTrue[1]/PosTrue[0]))
    print('the best ture rate is %d/%d-->%f' % (PosTrue[2], PosTrue[0], PosTrue[2]/PosTrue[0]))


def GetBestLevel(shapeletwordgroup, beta, mode=0):
    levelpattern = r'^\d+$'
    
    N = len(shapeletwordgroup['sampleidxs'][:])
    pos_num = int(N*beta)
    level = ''
    
    if mode == 0:
        bestscore = 0
        miniavgdist = float('inf')
        
        for levelkey in shapeletwordgroup.keys():
            if re.match(levelpattern, utilmx.Encode(levelkey)) is not None:
                dists = shapeletwordgroup[levelkey]['dists'][:]
                dists = np.sort(dists)
                # avgdist = sum(dists[:pos_num])/(pos_num - 1)
                # avgdist = avgdist/int(levelkey)
                avgdist = np.std(dists[:pos_num])
                score = shapeletwordgroup[levelkey]['score'][0]
                if score > bestscore:
                    bestscore = score
                    miniavgdist = avgdist
                    level = levelkey
                elif bestscore == score:
                    if avgdist < miniavgdist:
                        bestscore = score
                        miniavgdist = avgdist
                        level = levelkey
    
    return level
                
                    
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
                if utilmx.Encode(videolist[idx]) == videokey and rangelist[idx][0] == int(offset):
                    if data[0] == 1:
                        IDX[-1][2] = 1
                    # 确定该 annotation 在记录中的位置
                    IDX[-1][-1] = idx
                    break
    return IDX


def JudgeWhetherFound(annorange, locrange, sigma=0):
    # 判断两个区域范围的重叠区域是否大于一个比例阈值: sigma
    # 这里面存在一个问题： 覆盖应该是 标记的百分比， 还是shapelet的百分比
    a, b = annorange
    x, y = locrange
    # a, b = locrange
    # x, y = annorange
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
    recordfile = h5py.File(recordhdf5file, 'r')
    for word in recordfile.keys():
        # 获取当前文件的信息
        infokey = '%s/loginfo' % word
        idxkey = '%s/sampleidxs' % word
        infos = utilmx.Encode(recordfile[infokey][:][0])
        fx = float(re.findall(fxpattern, infos)[0])
        indexes = recordfile[idxkey][:]
        
        # according the above inforamtion to get the true info
        num = len(indexes)
        sample_indexes = worddict.ChooseSamples(word, fx, maxitems=num)
        clipidx = np.array([x[1:3] for x in sample_indexes if x[-1] == 1]).astype(np.int32)
        utilmx.WriteRecords2File(outfilepath, idxkey, clipidx, (num, 2), dtype=np.int32)
        
    recordfile.close()


def VoteForShapelet(shapeletwordgroup, rangelist, m):
    levelpattern = r'^\d+$'
    lengths = [int(x[1] - x[0]) for x in rangelist]
    DataMats = [np.zeros((m,)) for m in lengths]

    for levelkey in shapeletwordgroup.keys():
        if re.match(levelpattern, levelkey) is not None:
            locs = shapeletwordgroup[levelkey]['locs'][:]
            score = shapeletwordgroup[levelkey]['score'][0]
            # 进行 vote
            for idx, loc in enumerate(locs):
                DataMats[idx][loc:loc+int(levelkey)] += score
    
    locranges = np.zeros((len(rangelist), 2), dtype=np.uint8)
    # m = 30
    for idx, data in enumerate(DataMats):
        temp = np.zeros((lengths[idx]-m+1,))
        for i in range(len(temp)):
            temp[i] = sum(data[i:i+m])
        j = np.argmax(temp)
        locranges[idx] = np.array([j, j+m])

    return DataMats, locranges
    

def TestVoteinfluence(annotationfilepath, shapeletfilepath, m, alpha=0.3, sigma=0.5):
    annofile = h5py.File(annotationfilepath, mode='r')
    shapefile = h5py.File(shapeletfilepath, mode='r')
    
    words = list(annofile.keys())
    pos_true = [0, 0]
    for word in words:
        rangelist = shapefile[word]['sampleidxs'][:]
        videokeys = shapefile[word]['videokeys'][:]

        datamats, locranges = VoteForShapelet(shapefile[word], rangelist, m)
        IDxs = GetIdxofAnnotation(annofile[word], rangelist, videokeys)

        Posidxs = [x for x in IDxs if x[2] == 1]
        pos_ratio = len(Posidxs)/len(IDxs)
        if pos_ratio < alpha:
            continue
        
        pos_true[0] += 1
        recall = 0
        for item in Posidxs:
            videokey, offset, label, idx = item
            begin, end = annofile[word][videokey][offset][1:]
            lb, le = locranges[idx]
            if JudgeWhetherFound((begin, end), (lb, le), sigma=sigma) == 1:
                recall += 1
            
            # votedata = datamats[idx]
            # data = np.zeros_like(votedata)
            # data[int(begin):int(end)] += 4
            # plt.plot(votedata, 'b')
            # plt.plot(data, 'r')
            # plt.title('%s-%d' % (word, idx))
            # plt.show()
        if recall/len(Posidxs) > 0.5:
            pos_true[1] += 1
        # get the range according to the vote results
    print('the best ture rate is %d/%d-->%f' % (pos_true[1], pos_true[0], pos_true[1]/pos_true[0]))


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


def AugmentationScaleAnalysis(shapeletfilepath, augdatafilepath):
    pass
    augfile = h5py.File(augdatafilepath, 'r')
    shapeletfile = h5py.File(shapeletfilepath, 'r')

    wordlist = list(augfile.keys())

    for cword in wordlist:
        for levelkey in augfile[cword].keys():
            nums = 0
            for videokey in augfile[cword][levelkey].keys():
                nums += len(augfile[cword][levelkey][videokey][:])
            orinum = len(shapeletfile[cword][levelkey]['dists'][:])

            print('%s--%s: %d/%d = %f' % (cword, levelkey, nums, orinum, nums/orinum))
    

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
        # h5shapeletrecordED = '../data/spbsl/temprecord.hdf5'
        h5shapeletrecordED = '../data/spbsl/shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk_shapeletNetED.hdf5'
        
        CalculateRecallRate_h5file(annotationpath, h5shapeletrecordED, 0.5, sigma=0.5)
        # CalculateRecallRate_h5file(annotationpath, h5shapeletrecordNet, 0.4)
    
    elif testcode == 5:
        h5shapeletrecordED = '../data/spbsl/bk_shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk_shapeletNetED.hdf5'
        outh5shapeletrecordED = '../data/spbsl/bk1_shapeletED.hdf5'
        outh5shapeletrecordNet = '../data/spbsl/bk4_shapeletNetED.hdf5'

        # RecordReindex(h5shapeletrecordNet, worddictpath, subtitledictpath, outh5shapeletrecordNet)
        # CalculateRecallRate_h5file(annotationpath, outh5shapeletrecordED, threhold=0.5, sigma=0.5)
        for beta in np.arange(0.3, 1, 0.1):
            ResultsAnalysis(annotationpath, outh5shapeletrecordED, alpha=0.3, beta=beta, sigma=0.5)

    
    elif testcode == 6:
        h5shapeletrecordED = '../data/spbsl/bk1_shapeletED.hdf5'
        h5shapeletrecordNet = '../data/spbsl/bk3_shapeletNetED.hdf5'
        # DistAnalysis(h5shapeletrecordED, annotationpath)
        DistAnalysis(h5shapeletrecordNet, annotationpath)
    
    elif testcode == 7:
        h5shapeletrecordEDfilepath = '../data/spbsl/bk1_shapeletED.hdf5'
        augdatafilepath = '../data/spbsl/augdata.hdf5'

        AugmentationAnalysis(h5shapeletrecordEDfilepath, augdatafilepath)
    
    elif testcode == 8:
        shapeletfilepath = '../data/spbsl/bk2_shapeletED.hdf5'
        for m in range(13, 39):
            TestVoteinfluence(annotationpath, shapeletfilepath, m, alpha=0, sigma=0)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-t', '--testcode', type=int, default=8)
    Parser.add_argument('-s', '--server', action='store_true')

    args = Parser.parse_args()
    testcode = args.testcode
    server = args.server
    
    RunTest(testcode, server)