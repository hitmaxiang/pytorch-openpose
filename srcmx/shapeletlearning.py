'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-12-07 20:45:15
'''
import os
import time
import utilmx
import joblib
import tslearn
import torch
import h5py
import numpy as np
import shapeletmodel as SM
import PreprocessingData as PD
import matplotlib.pyplot as plt

from numba import jit
from utilmx import Records_Read_Write
from SubtitleDict import WordsDict, AnnotationDict


class ShapeletsFinding():
    def __init__(self, motion_dictpath, word_dictpath, subtitle_dictpath):
        # self.motiondatadict = joblib.load(motion_dictpath)
        self.motiondatafile = h5py.File(motion_dictpath, mode='r')
        self.cls_worddict = WordsDict(word_dictpath, subtitle_dictpath)
    
    def Getsamples(self, word):
        '''
        description: get the instance of the word, and random sample the negative samples
        param: word, the queried word
        return: pos_indexes, neg_indexes, pos_samples, neg_samples
        author: mario
        '''
        # 抽样得到 pos 以及 neg 的样本的索引以及clip位置
        # sample_indexes 的格式为：[videokey(str), begin, end, label]
        sample_indexes = self.cls_worddict.ChooseSamples(word, 1.5)
        samples = []

        # 从 motiondict 中 按照上面得到的索引位置提取数据
        # motiondata format: [motiondatas, scores]
        for i in range(len(sample_indexes)):
            videokey, beginindex, endindex = sample_indexes[i][:3]
            
            posedata = self.motiondatafile['posedata/pose/%s' % videokey][beginindex:endindex].astype(np.float32)
            handdata = self.motiondatafile['handdata/hand/%s' % videokey][beginindex:endindex].astype(np.float32)
            clip_data = np.concatenate((posedata, handdata), axis=1)

            # 针对每个 clip 数据, 只选取上面身的关节数据作为特征
            # clip_data = PD.MotionJointFeatures(clip_data, datamode='posehand', featuremode=0)
            clip_data = PD.MotionJointFeatures(clip_data, datamode='posehand', featuremode=0)
            clip_data = np.reshape(clip_data, (clip_data.shape[0], -1))
            # 因为原始的数据类型为int16， 在后续计算的过程中，容易溢出
            samples.append(clip_data)
        return samples, sample_indexes
    
    def train(self, word=None, method=2):

        if method == 1:
            self.current_shapelet_dict = utilmx.ShapeletRecords().ReadRecordInfo('../data/spbsl/shapeletED.rec')
            self.recodfilepath = '../data/spbsl/shapeletNetED.rec'
        elif method == 2:
            self.recodfilepath = '../data/spbsl/shapeletED.rec'
        else:
            self.recodfilepath = '../data/spbsl/shapeletany.rec'
        if word is None:
            words = self.cls_worddict.worddict.keys()
        elif isinstance(word, str):
            words = [word]
        elif isinstance(word, list):
            words = word
        minlen, maxlen, stride = 10, 30, 3

        if os.path.exists(self.recodfilepath):
            trainedrecords = utilmx.ReadShapeletRecords(self.recodfilepath)
        else:
            trainedrecords = {}

        for word in words:
            # 现阶段，对于sample特别多的先不分析
            if len(self.cls_worddict.worddict[word]) >= 500:
                continue
            if word in trainedrecords.keys():
                if len(trainedrecords[word]) >= int((maxlen-minlen)/stride):
                    continue
            self.word = word
            samples, sample_indexes = self.Getsamples(word)

            # write the base line info
            with open(self.recodfilepath, 'a') as f: 
                for info in sample_indexes:
                    if sample_indexes[-1] == 1:
                        f.write('%s-%d\t' % (info[0], info[1]))
                f.write('\n')
            
            for m in range(minlen, maxlen, stride):
                if word in trainedrecords.keys():
                    if m in trainedrecords[word]:
                        continue
                if method == 0:
                    # self.FindShaplets_dtw_methods(samples, sample_indexes, m)

                    pass
                elif method == 1:
                    self.FindShaplets_Net_ED(samples, sample_indexes, m)
                    pass
                elif method == 2:
                    # self.FindShaplets_brute_force_ED(samples, sample_indexes, m)
                    self.FindShaplets_brute_force_ED(samples, sample_indexes, m)
    
    def FindShaplets_brute_force_ED(self, samples, sample_indexes, m_len):
        
        begin_time = time.time()
        shapeletmodel = SM.ShapeletMatrixModel()
        BestKshapelets = utilmx.Best_K_Items(K=10)

        # 对样本集合进行归一化处理
        samples = PD.NormlizeData(samples, mode=1)
        
        labels = torch.tensor([x[-1] for x in sample_indexes])
        shapeletmodel.train(samples, labels, m_len)

        # draw the result
        locs = shapeletmodel.locs
        shapindex = shapeletmodel.shapeindex
        dis = shapeletmodel.dis
        score = shapeletmodel.score

        key = '%s-framindex:%d-offset:%d-m_len:%d' % (sample_indexes[shapindex][0],
                                                      sample_indexes[shapindex][1],
                                                      locs[shapindex],
                                                      m_len)

        validloc = [loc for i, loc in locs if sample_indexes[i][-1] == 1]

        BestKshapelets.insert(score, [key, validloc])
        # shapelet = samples[shapindex][locs[shapindex]:locs[shapindex]+m_len, 0]
        Headerinfo = 'the word:%s with m length: %d' % (self.word, m_len)
        BestKshapelets.wirteinfo(Headerinfo, self.recodfilepath, 'a')

        # samplenum = len(samples)
        # Y = [x[-1] for x in sample_indexes]
        # plt.switch_backend('agg')

        # plt.figure(0)
        # for i in range(samplenum):
        #     if Y[i] == 1:
        #         plt.scatter([i], [dis[i].item()], c='r', marker='o')
        #     else:
        #         plt.scatter([i], [dis[i].item()], c='g', marker='*')
        # plt.savefig('../data/spbsl/img/%s-%d-dis.jpg' % (self.word, m_len))
        print('the %d word %s of %d length cost %f seconds with score %f' % (len(samples)//2,
                                                                             self.word,
                                                                             m_len,
                                                                             time.time()-begin_time,
                                                                             score))

    def FindShaplets_Net_ED(self, samples, sample_indexes, m_len):
        # 对样本集合进行归一化处理
        samples = PD.NormlizeData(samples, mode=1)

        BestKshapelets = utilmx.Best_K_Items(K=10)

        lenghts = [len(x) for x in samples]
        N, T, D = len(samples), max(lenghts), samples[0].shape[1]
        # 构建训练样本，不足的取零
        X = torch.zeros(N, D, T, dtype=torch.float32)
        for i in range(N):
            X[i, :, :lenghts[i]] = torch.from_numpy(samples[i]).permute(1, 0)
        Y = torch.tensor([x[-1] for x in sample_indexes])
        
        # Set the default query
        valid = False
        try:
            shapelet = self.current_shapelet_dict[self.word][str(m_len)]['shapelet']
            videonum, frameindex, offset, length, score = shapelet[0]

            for i in range(len(sample_indexes)):
                if sample_indexes[i][-1] == 1:
                    videokey, beginindex = sample_indexes[i][:2]
                    if videokey == videonum and beginindex == frameindex:
                        print('the default shapelt will be initialized with the record')
                        valid = True
                        break
        except Exception:
            print('the %s is not in the shapeletED record')
        if valid is True:
            index = i
            bindex = offset
        else:
            index = 0
            bindex = 127
        
        eindex = bindex + m_len
        default_query = X[index, :, bindex:eindex].unsqueeze(0)
        Net = SM.ShapeletNetModel(shape=(m_len, D), query=default_query)
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()
            Net = Net.cuda()

        Net.train(X, Y)

        dis, locs, score = Net(X, Y)
        # loc, dic = Net.localizeshape(X)
        # print(loc)
        if score > 0:
            dis = dis.cpu().numpy()
            locs = locs.cpu().numpy()

            key = '%s-framindex:%d-offset:%d-m_len:%d' % (sample_indexes[0][0],
                                                          sample_indexes[0][1],
                                                          locs[0],
                                                          m_len)

            BestKshapelets.insert(score, [key, locs])
            # shapelet = samples[shapindex][locs[shapindex]:locs[shapindex]+m_len, 0]
            Headerinfo = 'the word:%s with m length: %d' % (self.word, m_len)
            BestKshapelets.wirteinfo(Headerinfo, self.recodfilepath, 'a')

        
def Test(testcode):
    motionhdf5filepath = '../data/spbsl/motiondata.hdf5'
    # motionsdictpath = '../data/spbsl/motionsdic.pkl'
    worddictpath = '../data/spbsl/WordDict.pkl'
    subtitledictpath = '../data//spbsl/SubtitleDict.pkl'
    # annotationdictpath = '../data/annotationdict.pkl'
    if testcode == 0:
        cls_shapelet = ShapeletsFinding(motionhdf5filepath, worddictpath, subtitledictpath)
        # consider: 500, thank:2153, supermarket:60, weekend: 76, expert: 99
        # cls_shapelet.train('supermarket', method=1)
        cls_shapelet.train(method=1)

        
if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    Test(0)