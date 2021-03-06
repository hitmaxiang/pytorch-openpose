'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-01-11 16:33:19
LastEditors: mario
LastEditTime: 2021-03-06 22:33:36
'''
import os
import re
import time
import math
import torch
import h5py
import utilmx
import PreprocessingData as PD
import numpy as np


class DataAugmentation():
    def __init__(self, shapeletrecordpath, outpathpath, motiondatapath, MaxSegLength):
        # self.shapeletdict = utilmx.ShapeletRecords().ReadRecordInfo(shapeletrecordpath)
        # 现在的shapelet 具有的是 hdf5 的格式
        self.shapeletfile = h5py.File(shapeletrecordpath, mode='r')
        self.h5motionfile = h5py.File(motiondatapath, mode='r')
        self.outputpath = outpathpath
        self.MaxSegLength = MaxSegLength
    
    def GetShapeletPattern(self, word, mode=0, scale=0.3):
        # 根据 Word 的 record 信息确定模板以及对饮的阈值
        shapelets = []
        mlenpattern = r'^\d+$'
        # mode 0: 直接返回每个 shapelet, 并将 scale 之内的最大距离作为 阈值 sigma
        if mode == 0:
            for key in self.shapeletfile[word].keys():
                if re.match(mlenpattern, key) is None:
                    continue
                # 针对一个 mlen 级别， 获取它的 shaplet, 并分析得到它的距离阈值
                basekey = '%s/%s' % (word, key)
                shapelet = self.shapeletfile[basekey]['shapelet'][:]
                dists = self.shapeletfile[basekey]['dists'][:]

                dists = np.sort(dists)
                sigma = dists[int(len(dists) * scale)]
                
                shapelets.append((key, shapelet, sigma))
        # mode 1: 返回 一个最优的 shapelet
        elif mode == 1:
            # 选择最好的区分度，当区分度一样的话，选择最小的平均距离
            bestshaplet = {'score': 0, 'avgdist': float('inf'), 'shapelet': None}
            for key in self.shapeletfile[word].keys():
                if re.match(mlenpattern, key) is None:
                    continue
                basekey = '%s/%d' % (word, key)
                shapelet = self.shapeletfile[basekey]['shapelet'][:]
                dists = self.shapeletfile[basekey]['dists'][:]
                score = self.shapeletfile[basekey]['score'][:]

                dists = np.sort(dists)
                avgdist = dists[:int(len(dists) * scale)]
                sigma = dists[int(len(dists) * scale)]
                if score > bestshaplet['score']:
                    temp = (key, shapelet, sigma)
                    bestshaplet['score'] = score
                    bestshaplet['avgdist'] = avgdist
                    bestshaplet['shapelet'] = temp
                elif score == bestshaplet['score'] and bestshaplet['avgdist'] > avgdist:
                    temp = (key, shapelet, sigma)
                    bestshaplet['score'] = score
                    bestshaplet['avgdist'] = avgdist
                    bestshaplet['shapelet'] = temp
                    
            shapelets.append(bestshaplet['shapelet'])
        
        return shapelets

    def AugmentateOneShapelet(self, word, shapeletpattern, sigma):
        levelkey, shapelet, sigma = shapeletpattern
        # 获取 shapelet 的特征类型以及细节
        infopattern = r'fx:(.+)-datamode:(.+)-featuremode:(\d+)$'
        ranges = self.shapeletfile[word]['sampleidxs'][:]
        videokeys = self.shapeletfile[word]['videokeys'][:]
        locs = self.shapeletfile[word][levelkey]['locs'][:]
        info = self.shapeletfile[word]['loginfo'][0]
        fx, datamode, featuremode = re.findall(infopattern, info)[0]
        featuremode = int(featuremode)
        
        # 这种情况下，是 shapletfinding 的结果
        if len(shapelet) == 1:  
            index = shapelet[0]
            minid, maxid = ranges[index]
            begidx = locs[index] + minid
            endidx = begidx + int(levelkey)
            videokey = videokeys[index]

            posedata = self.h5motionfile['posedata/pose/%s' % videokey][begidx:endidx].astype(np.float32)
            handdata = self.h5motionfile['handdata/hand/%s' % videokey][begidx:endidx].astype(np.float32)
            shapeletdata = np.concatenate((posedata, handdata), axis=1)

            shapeletdata = PD.MotionJointFeatures(shapeletdata, datamode, featuremode)
            shapeletdata = np.reshape(shapeletdata, (shapeletdata.shape[0], -1))
        else:
            shapeletdata = shapelet
        
        AugmentDict = {}
        for videokey in self.h5motionfile['posedata/pose'].keys():
            AugmentDict[videokey] = []

            N = len(self.h5motionfile['posedata/pose/%s' % videokey][:])
            m = len(shapeletdata)
            SegNum = math.ceil(N/self.MaxSegLength)
            begtime = time.time()
            for ite in range(SegNum):
                begidx = max(ite * self.MaxSegLength - m, 0)
                endidx = min((ite+1)*self.MaxSegLength, N)

                posedata = self.h5motionfile['posedata/pose/%s' % videokey][begidx:endidx]
                handdata = self.h5motionfile['handdata/hand/%s' % videokey][begidx:endidx]

                segdata = np.concatenate((posedata, handdata), axis=1).astype(np.float32)
            
                # 对原始的动作数据进行特征提取处理
                segdata = PD.MotionJointFeatures(segdata, datamode, featuremode)
                segdata = np.reshape(segdata, (segdata.shape[0], -1))
                
                if torch.cuda.is_available():
                    pattern = torch.from_numpy(shapeletdata).cuda()
                    segdata = torch.from_numpy(segdata).cuda()
                    with torch.no_grad():
                        dist = utilmx.SlidingDistance_torch(pattern, segdata)
                else:
                    dist = utilmx.SlidingDistance(shapeletdata, segdata)
                preidx = None
                for idex, dis in enumerate(dist):
                    if dis < sigma:
                        if preidx is not None:
                            if idex - preidx > m/2:
                                AugmentDict[videokey].append((begidx+idex, dis))
                                preidx = idex
                            elif dis < AugmentDict[videokey][-1][1]:
                                AugmentDict[videokey][-1] = (begidx+idex, dis)
                                preidx = idex
                        else:
                            AugmentDict[videokey].append((begidx+idex, dis))
                            preidx = idex
                print('%f seconds-->%s:%d-%d/%d' % (time.time() - begtime, videokey, begidx, endidx, N))
                begtime = time.time()
        return AugmentDict

    def DataAugmentate(self, word=None, mode=0, scale=0.3, overwrite=False):
        # 为特定的Word， 如果不给定的话， 默认将为 record 中的所有Word进行
        if isinstance(word, list):
            wordlist = word
        elif isinstance(word, str):
            wordlist = [word]
        elif word is None:
            wordlist = list(self.shapeletfile.keys())
        else:
            print('please input the word or wordlist with correct form')
            return
            
        h5outfile = h5py.File(self.outputpath, mode='w')
        for cword in wordlist:
            shapeletpatterns = self.GetShapeletPattern(cword, mode, scale)
            for shapeletpattern in shapeletpatterns:
                levelkey, shapelet, sigma = shapeletpattern
                groupkey = '%s/%s' % (cword, levelkey)
                if overwrite is False:
                    if groupkey in h5outfile.keys():
                        continue
                # 如果已经 augdata 已经存在的话， 将会删除现存的数据
                elif groupkey in h5outfile.keys():
                    del h5outfile[groupkey]
                
                # 注意的一点是，这里给定的shapelet data 已经是提取 动作特征之后的了
                augmentdict = self.AugmentateOneShapelet(cword, shapeletpattern, sigma)
                for videokey in augmentdict.keys():
                    datakey = '%s/%s' % (groupkey, videokey)
                    data = np.array(augmentdict[videokey])
                    h5outfile.create_dataset(datakey, data=data)


if __name__ == '__main__':
    shapletfilepath = '../data/spbsl/bk1_shapeletED.hdf5'
    motiondatapath = '../data/spbsl/motiondata.hdf5'
    outfilepath = '../data/spbsl/augdata.hdf5'
    Augmentation = DataAugmentation(shapletfilepath, outfilepath, motiondatapath, MaxSegLength=100000)
    Augmentation.DataAugmentate()