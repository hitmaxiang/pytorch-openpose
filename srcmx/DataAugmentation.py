'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-01-11 16:33:19
LastEditors: mario
LastEditTime: 2021-01-11 21:59:50
'''
import os
import math
import torch
import h5py
import utilmx
import PreprocessingData as PD
import numpy as np


class DataAugmentation():
    def __init__(self, shapeletrecordpath, outpathpath, motiondatapath, MaxSegLength):
        self.shapeletdict = utilmx.ShapeletRecords().ReadRecordInfo(shapeletrecordpath)
        self.h5motionfilepath = motiondatapath
        self.outputpath = outpathpath
        self.MaxSegLength = MaxSegLength
    
    def GetShapeltPattern(self, word, mode):
        # 根据 Word 的 record 信息确定模板以及对饮的阈值
        pass

    def AugmentateOneShapelet(self, shapeletdata, sigma):
        # 根据现有的 shapelet 从全局的 motiondata 中搜寻满足小于 sigma 的所有候选
        h5motionfile = h5py.File(self.h5motionfilepath, mode='r')
        
        AugmentDict = {}

        for videokey in h5motionfile.keys():
            AugmentDict[videokey] = []

            posedata = h5motionfile['posedata/pose/%s' % videokey]
            handdata = h5motionfile['handdata/hand/%s' % videokey]

            clipdata = np.concatenate((posedata, handdata), axis=1)

            N, m = len(clipdata), len(shapeletdata)
            
            SegNum = math.ceil(N/self.MaxSegLength)

            for ite in range(SegNum):
                begidx = max(ite * self.MaxSegLength - m, 0)
                endidx = min((ite+1)*self.MaxSegLength, N)

                segdata = clipdata[begidx:endidx].astype(np.float32)
                # 对原始的动作数据进行特征提取处理
                segdata = PD.MotionJointFeatures(segdata, datamode='posehand', featuremode=1)
                segdata = np.reshape(segdata, (segdata.shape[0], -1))
                
                dist = utilmx.SlidingDistance_torch(shapeletdata, segdata)
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
                print('%s:%d-%d/%d' % (videokey, begidx, endidx, N))

    def DataAugmentate(self, word=None, mode=0, overwrite=False):
        # 为特定的Word， 如果不给定的话， 默认将为 record 中的所有Word进行
        if word is None:
            wordlist = [word]
        else:
            wordlist = list(self.shapeletdict.keys())
        
        h5outfile = h5py.File(self.outputpath, 'a')
        for cword in wordlist:
            shapletpatterns = self.GetShapeltPattern(cword, mode)
            for levelkey, shapelet, sigma in shapletpatterns:
                groupkey = '%s/%s' % (cword, levelkey)
                if overwrite is False:
                    if groupkey in h5outfile.keys():
                        continue
                
                # 如果已经 augdata 已经存在的话， 将会删除现存的数据
                if groupkey in h5outfile.keys():
                    del h5outfile[groupkey]
                
                # 注意的一点是，这里给定的shapelet data 已经是提取 动作特征之后的了，而且
                augmentdict = self.AugmentateOneShapelet(shapelet, sigma)
                for videokey in augmentdict.keys():
                    datakey = '%s/%s' % (groupkey, videokey)
                    data = np.array(augmentdict[videokey])
                    h5outfile.create_dataset(datakey, data=data)

        
        


    

    

                            


                    

        
    
