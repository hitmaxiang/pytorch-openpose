'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-01-11 16:33:19
LastEditors: mario
LastEditTime: 2021-03-10 23:35:39
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
                basekey = '%s/%s' % (word, key)
                shapelet = self.shapeletfile[basekey]['shapelet'][:]
                dists = self.shapeletfile[basekey]['dists'][:]
                score = self.shapeletfile[basekey]['score'][0]

                dists = np.sort(dists)
                validnum = int(len(dists) * scale)
                avgdist = np.sum(dists[:validnum])/(validnum - 1)
                sigma = dists[validnum-1]
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
            begidx, endidx = ranges[index]
            loc = locs[index]
            videokey = videokeys[index]

            posedata = self.h5motionfile['posedata/pose/%s' % videokey][begidx:endidx].astype(np.float32)
            handdata = self.h5motionfile['handdata/hand/%s' % videokey][begidx:endidx].astype(np.float32)
            shapeletdata = np.concatenate((posedata, handdata), axis=1)

            shapeletdata = PD.MotionJointFeatures(shapeletdata, datamode, featuremode)
            shapeletdata = np.reshape(shapeletdata, (shapeletdata.shape[0], -1))
            shapeletdata = shapeletdata[loc:loc+int(levelkey)]
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
                
                pattern = torch.from_numpy(shapeletdata)
                segdata = torch.from_numpy(segdata)
                if torch.cuda.is_available():
                    pattern = pattern.cuda()
                    segdata = segdata.cuda()
                    with torch.no_grad():
                        dist = utilmx.SlidingDistance_torch(pattern, segdata)
                        # dist = dist.cpu()
                else:
                    dist = utilmx.SlidingDistance_torch(pattern, segdata)
                preidx = None
                # print(min(dist))
                for idex, dis in enumerate(dist):
                    if dis < sigma:
                        if preidx is not None:
                            if idex - preidx > m/2:
                                AugmentDict[videokey].append((begidx+idex, dis.item()))
                                preidx = idex
                            elif dis < AugmentDict[videokey][-1][1]:
                                AugmentDict[videokey][-1] = (begidx+idex, dis.item())
                                preidx = idex
                        else:
                            AugmentDict[videokey].append((begidx+idex, dis.item()))
                            preidx = idex
            # print('%f seconds-->%s:%d-%d/%d' % (time.time() - begtime, videokey, begidx, endidx, N))
            print('the sliding distance of %s with length %d calculate %f seconds' % (videokey, N, time.time()-begtime))
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
            
        h5outfile = h5py.File(self.outputpath, mode='a')
        for i, cword in enumerate(wordlist):
            shapeletpatterns = self.GetShapeletPattern(cword, mode, scale)
            for shapeletpattern in shapeletpatterns:
                begtime = time.time()
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
                    h5outfile.flush()
                print('write the data of %s with %f seconds' % (groupkey, time.time() - begtime))
        h5outfile.close()


class FastDataAugmenter():
    def __init__(self, motionfilepath, shapeletfilepath, scale=0.3, maxseglen=10000):
        self.motionfilepath = motionfilepath
        self.shapeletfilepath = shapeletfilepath
        self.scale = scale
        self.maxseglen = maxseglen
        self.featuredatapath = '..data/spbsl/tempfeaturedata.hdf5'
    
    def PrepareGlobalData(self, datamode, featuremode, rewrite=False):
        # 因为每个shapelet 都要和全部的数据进行比对，为了加速计算，对于motiondata的预处理可以先进行
        # 数据的格式应该是 featuremode/videokey ==> [TxD]
        motionfile = h5py.File(self.motionfilepath, 'r')
        videokeys = motionfile['posedata/pose'].keys()

        # 判断哪些数据是需要新添加的
        writekeys = []
        with h5py.File(self.featuredatapath, mode='r') as f:
            for videokey in videokeys:
                key = '%s-%d/%s' % (datamode, featuremode, videokey)
                if rewrite is True or key not in f.keys():
                    writekeys.append(key)
        
        if len(writekeys) > 0:
            print('prepare the motiondata with datamode:%s, featuremode:%d' % (datamode, featuremode))

        for key in writekeys:
            videokey = key.split('/')[1]
            posedata = motionfile['posedata/pose/%s' % videokey][:].astype(np.float32)
            handdata = motionfile['handdata/hand/%s' % videokey][:].astype(np.float32)
            fram_num = len(posedata)
            data = np.concatenate((posedata, handdata), axis=1)
            data = PD.MotionJointFeatures(data, datamode=datamode, featuremode=featuremode)
            data = np.reshape(data, (data.shape[0], -1))

            utilmx.WriteRecords2File(self.featuredatapath, key, data, (fram_num,), dtype=np.float32)
        
        motionfile.close()
    
    def SlidingSearching(self, shapelets, datamode, featuremode):
        featurekey = '%s-%d' % (datamode, featuremode)
        premotionfile = h5py.File(self.featuredatapath, 'r')
        
        for videokey in premotionfile[featurekey].keys():
            videodata = premotionfile[featurekey][videokey][:]
            videodata = torch.from_numpy(videodata)
            if torch.cuda.is_available():
                videodata.cuda()
        
            for shapelet in shapelets:
                word, shapeletdata, sigma = shapelet
                levelkey = str(len(shapeletdata))
                shapeletdata = torch.from_numpy(shapeletdata)
                if torch.cuda.is_available():
                    shapeletdata = shapeletdata.cuda()
                
        
        


    def GetShapeletpatterns(self, shapeletfilepath, method, scale, words=None):
        # 从shapeletfile 中提取作为核心的 shaplet, 具有不同的方式，以及精纯度
        # 一般建议将scale选择较大的， 然后在得到的结果中根据需要进行不同的筛选
        levelpattern = r'^\d+$'
        infopattern = r'datamode:(.+)-featuremode:(\d+)$'
        
        ShapePatternDict = {}

        shapeletfile = h5py.File(shapeletfilepath, 'r')
        # 当不给定 words 的时候，默认是所有的 Word
        if words is None:
            words = list(shapeletfile.keys())

        for word in words:
            if word not in shapeletfile.keys():
                continue

            # 获取该 Word 的 shapelet 的信息
            ranges = shapeletfile[word]['sampleidxs'][:]
            videokeys = shapeletfile[word]['videokeys'][:]
            info = shapeletfile[word]['loginfo'][0]
            datamode, featuremode = re.findall(infopattern, info)[0]
            
            self.PrepareGlobalData(datamode, int(featuremode))

            featurekey = '%s-%s' % (datamode, featuremode)
            if featurekey not in ShapePatternDict.keys():
                ShapePatternDict[featurekey] = []
            
            # 返回所有找到的shapelet， 各个级别的都要
            if method == 0:
                for levelkey in shapeletfile[word].keys():
                    if re.match(levelpattern, levelkey) is None:
                        continue
                    shapelet = shapeletfile[word][levelkey]['shapelet'][:]
                    dists = np.sort(shapeletfile[word][levelkey]['dists'][:])
                    sigma = dists[int(len(dists)*scale)]
                    # shapelet finding 的结果
                    if len(shapelet) == 1:
                        index = shapelet[0]
                        loc = shapeletfile[word][levelkey]['locs'][index]
                        videokey = videokeys[index]
                        begidx = ranges[index][0] + loc
                        
                        # 从该特征下的预处理motiondata中直接提取片段
                        with h5py.File(self.featuredatapath, 'r') as f:
                            shapeletdata = f[featurekey][videokey][begidx:begidx+int(levelkey)]
                        
                        ShapePatternDict[featurekey].append((word, shapeletdata, sigma))
                    
                    #  the results of shapelet learning 
                    else:
                        shapeletdata = shapelet
                        ShapePatternDict[featurekey].append((word, shapeletdata, sigma))
            
            # 对于每个 word 来说, 筛选一个最优的shapelet作为搜索的模板
            # 这里的最优： score 最高， 然后相对更小的平均距离
            elif method == 1:
                # 定义最优的 shapeletinfo
                bestshaplet = {'score': 0, 'avgdist': float('inf'), 'shapeletinfo': None}
                for levelkey in shapeletfile[word].keys():
                    if re.match(levelpattern, levelkey) is None:
                        continue
                    shapelet = shapeletfile[word][levelkey]['shapelet'][:]
                    score = self.shapeletfile[word][levelkey]['score'][0]
                    dists = np.sort(shapeletfile[word][levelkey]['dists'][:])

                    validnum = int(len(dists) * scale)
                    sigma = dists[validnum]
                    avgdist = np.sum(dists[:validnum+1])/validnum

                    if score > bestshaplet['score']:
                        temp = (levelkey, shapelet, sigma)
                        bestshaplet['score'] = score
                        bestshaplet['avgdist'] = avgdist
                        bestshaplet['shapeletinfo'] = temp
                    elif score == bestshaplet['score'] and bestshaplet['avgdist'] > avgdist:
                        temp = (levelkey, shapelet, sigma)
                        bestshaplet['score'] = score
                        bestshaplet['avgdist'] = avgdist
                        bestshaplet['shapeletinfo'] = temp
                
                levelkey, shapelet, sigma = bestshaplet['shapeletinfo']
                # the result of shapelet finding
                if len(shapelet) == 1:
                    index = shapelet[0]
                    loc = shapeletfile[word][levelkey]['locs'][index]
                    videokey = videokeys[index]
                    begidx = ranges[index][0] + loc
                    
                    # 从该特征下的预处理motiondata中直接提取片段
                    with h5py.File(self.featuredatapath, 'r') as f:
                        shapeletdata = f[featurekey][videokey][begidx:begidx+int(levelkey)]
                    
                    ShapePatternDict[featurekey].append((word, shapeletdata, sigma))
                
                #  the results of shapelet learning 
                else:
                    shapeletdata = shapelet
                    ShapePatternDict[featurekey].append((word, shapeletdata, sigma))

        return ShapePatternDict









            
        
if __name__ == '__main__':
    shapletfilepath = '../data/spbsl/shapeletED.hdf5'
    motiondatapath = '../data/spbsl/motiondata.hdf5'
    outfilepath = '../data/spbsl/augdata.hdf5'
    Augmentation = DataAugmentation(shapletfilepath, outfilepath, motiondatapath, MaxSegLength=100000)
    Augmentation.DataAugmentate(mode=1, overwrite=True)