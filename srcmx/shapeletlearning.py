'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2021-01-11 23:09:13
'''
import os
import time
import utilmx
import joblib
import tslearn
import torch
import h5py
import argparse
import numpy as np
import PaperExperiments as PE
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

        # 数据特征类型
        self.datamode = 'posehand'
        self.featuremode = 1
        self.normmode = False

        # m_len level
        self.min_m = 15
        self.max_m = 40
        self.stride_m = 2

        # 选取样本时候的拓宽范围
        self.delayfx = 1.5

        # 默认的输出存储文件
        self.shapeletEDfile = '../data/spbsl/shapeletED.hdf5'
        self.shapeletNetfile = '../data/spbsl/shapeletNetED.hdf5'
    
    def SetDataMode(self, **args):
        modevaluedict = {'datamode': self.datamode, 'featuremode': self.featuremode,
                         'normmode': self.normmode, 'min_m': self.min_m,
                         'max_m': self.max_m, 'stride_m': self.stride_m,
                         'delayfx': self.delayfx, 'shapeletEDfile': self.shapeletEDfile,
                         'shapeletNetfile': self.shapeletNetfile}
                         
        definedkeys = modevaluedict.keys()
        for key, value in args:
            if key in definedkeys:
                modevaluedict[key] = value

    def Getsamples(self, word):
        '''
        description: get the instance of the word, and random sample the negative samples
        param: word, the queried word
        return: pos_indexes, neg_indexes, pos_samples, neg_samples
        author: mario
        '''
        # 抽样得到 pos 以及 neg 的样本的索引以及clip位置
        # sample_indexes 的格式为：[videokey(str), begin, end, label]
        sample_indexes = self.cls_worddict.ChooseSamples(word, self.delayfx, maxitems=300)
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
            clip_data = PD.MotionJointFeatures(clip_data, datamode=self.datamode, featuremode=self.featuremode)
            clip_data = np.reshape(clip_data, (clip_data.shape[0], -1))
            # 因为原始的数据类型为int16， 在后续计算的过程中，容易溢出
            samples.append(clip_data)
        return samples, sample_indexes
    
    def WriteRecords2File(self, key, data, shape, dtype):
        with h5py.File(self.h5recordpath, 'a') as f:
            if key in f.keys():
                if f[key][:].shape != shape:
                    del f[key]
                    f.create_dataset(key, shape, dtype=dtype)
            else:
                f.create_dataset(key, shape, dtype=dtype)
            f[key][:] = data
    
    def train(self, word=None, method=2, h5recordpath=None, overwrite=False):
        # 确定最终的存储文件
        if h5recordpath is not None:
            self.h5recordpath = h5recordpath
        elif method == 1:
            self.h5recordpath = self.shapeletNetfile
        elif method == 2:
            self.h5recordpath = self.shapeletEDfile

        self.overwrite = overwrite
        
        # 确定要提取的 given Word
        if word is None:
            words = self.cls_worddict.worddict.keys()
        elif isinstance(word, str):
            words = [word]
        elif isinstance(word, list):
            words = word

        minlen, maxlen, stride = self.min_m, self.max_m, self.stride_m

        for word in words:
            # 现阶段，对于sample特别多的先不分析
            # if len(self.cls_worddict.worddict[word]) >= 500:
            #     continue
            
            write_mlen_list = []

            with h5py.File(self.h5recordpath, 'a') as f:
                if word in f.keys() and self.overwrite is True:
                    del f[word]
                # 判断要写的是否已经在文件中了
                for m_len in range(minlen, maxlen, stride):
                    groupkey = '%s/%d' % (word, m_len)
                    if groupkey not in f.keys():
                        write_mlen_list.append(m_len)
            
            # 如果都在文件中的话就进入下一个 Word， 否则的话进入到 shapelet 学习的阶段
            if len(write_mlen_list) == 0:
                continue
            
            self.word = word
            samples, sample_indexes = self.Getsamples(word)
            # 为了保证完整的信息，这里将会把samples 准确的位置信息进行记录
            posidxs = [x[:3] for x in sample_indexes if x[-1] == 1]
            videokeys = [x[0] for x in posidxs]
            clipidx = np.array([x[1:] for x in posidxs]).astype(np.int16)

            strdt = h5py.string_dtype(encoding='utf-8')
            pos_num = len(videokeys)

            # 写入每个 sample 的起始范围
            idxkey = '%s/sampleidxs' % word
            self.WriteRecords2File(idxkey, clipidx, (pos_num, 2), dtype=np.int32)
            # 写入每个 sample 所在的 videokey
            vdokey = '%s/videokeys' % word
            self.WriteRecords2File(vdokey, videokeys, (pos_num, ), dtype=strdt)
            # 写入 word 的 loginfo
            infokey = '%s/loginfo' % word
            infomsg = 'fx:%.2f-datamode:%s-featuremode:%d' % (self.delayfx, self.datamode, self.featuremode)
            self.WriteRecords2File(infokey, infomsg, (1, ), dtype=strdt)

            # Do the training loop
            for m_len in write_mlen_list:

                if method == 1:
                    # using the shapeletnet to learn the shapelet
                    self.FindShaplets_Net_ED(samples, sample_indexes, m_len)
                
                elif method == 2:
                    # using the matrix brute force to find the shapelet
                    self.FindShaplets_brute_force_ED(samples, sample_indexes, m_len)
    
    # 使用蛮力 matrix profile 的方式进行 shapelet 的 finding
    def FindShaplets_brute_force_ED(self, samples, sample_indexes, m_len):
        begin_time = time.time()
        shapeletmodel = SM.ShapeletMatrixModel()

        # 对样本集合进行归一化处理
        if self.normmode:
            samples = PD.NormlizeData(samples, mode=1)
        
        labels = torch.tensor([x[-1] for x in sample_indexes])
        shapeletmodel.train(samples, labels, m_len)

        # get the training results
        locs, dists = [], []
        for i in range(len(sample_indexes)):
            if sample_indexes[i][-1] == 1:
                locs.append(shapeletmodel.locs[i])
                dists.append(shapeletmodel.dis[i])
        
        locs = np.array(locs).astype(np.int16)
        dists = np.array(dists).astype(np.float32)
        shapelet = np.array([shapeletmodel.shapeindex]).astype(np.int16)
        score = np.array([shapeletmodel.score]).astype(np.float32)

        # construct the database keyword
        basekey = '%s/%d/' % (self.word, m_len)
        locskey = basekey + 'locs'
        shapletkey = basekey + 'shapelet'
        distkey = basekey + 'dists'
        scorekey = basekey + 'score'

        pos_num = len(locs)
        # write the results into the recordfile
        self.WriteRecords2File(locskey, locs, (pos_num, ), dtype=np.int16)
        self.WriteRecords2File(distkey, dists, (pos_num, ), dtype=np.float32)
        self.WriteRecords2File(shapletkey, shapelet, (1, ), dtype=np.int16)
        self.WriteRecords2File(scorekey, score, (1, ), dtype=np.float32)

        print('the %d word %s of %d length cost %f seconds with score %f' % (len(samples)//2,
                                                                             self.word,
                                                                             m_len,
                                                                             time.time()-begin_time,
                                                                             score[0]))

    # 使用 shapeletlearning net 的方式进行 shapelet 的 learning
    def FindShaplets_Net_ED(self, samples, sample_indexes, m_len):
        # 对样本集合进行归一化处理
        if self.normmode:
            samples = PD.NormlizeData(samples, mode=1)

        # BestKshapelets = utilmx.Best_K_Items(K=10)

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
            # 根据 brute force 的结果为 net 的参数进行初始化
            if os.path.exists(self.shapeletEDfile):
                with h5py.File(self.shapeletEDfile, 'r') as f:
                    locskey = '%s/%d/locs' % (self.word, m_len)
                    shapeidxkey = '%s/%d/shapelet' % (self.word, m_len)
                    idxkey = '%s/sampleidxs' % self.word
                    vdokey = '%s/videokeys' % self.word
                    # 获取 found shapelet information
                    shapeletidx = f[shapeidxkey][0]
                    offset = f[locskey][shapeletidx]
                    videonum = f[vdokey][shapeletidx]
                    frameindex = f[idxkey][shapeletidx, 0]

            for i in range(len(sample_indexes)):
                if sample_indexes[i][-1] == 1:
                    videokey, beginindex = sample_indexes[i][:2]
                    if videokey == videonum and beginindex == frameindex:
                        print('the default shapelt will be initialized with the record')
                        valid = True
                        break
        except Exception:
            print('the %s is not in the %s file' % self.shapeletEDfile)
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

        # get the trained data
        distance, location, score = Net(X, Y)
        distance = distance.cpu().numpy()
        location = location.cpu().numpy()
        shapelet = Net.query.detach().cpu().numpy()
        
        locs, dists = [], []
        for i in range(len(sample_indexes)):
            if sample_indexes[i][-1] == 1:
                locs.append(location[i])
                dists.append(distance[i])
        
        locs = np.array(locs).astype(np.int16)
        dists = np.array(dists).astype(np.float32)
        shapelet = shapelet.astype(np.float32)
        score = np.array([score]).astype(np.float32)

        # construct the database keyword
        basekey = '%s/%d/' % (self.word, m_len)
        locskey = basekey + 'locs'
        shapletkey = basekey + 'shapelet'
        distkey = basekey + 'dists'
        scorekey = basekey + 'score'

        pos_num = len(locs)
        # write the results into the recordfile
        self.WriteRecords2File(locskey, locs, (pos_num, ), dtype=np.int16)
        self.WriteRecords2File(distkey, dists, (pos_num, ), dtype=np.float32)
        self.WriteRecords2File(shapletkey, shapelet, shapelet.shape, dtype=np.float32)
        self.WriteRecords2File(scorekey, score, (1, ), dtype=np.float32)


def RunTest(testcode, method, retrain):
    motionhdf5filepath = '../data/spbsl/motiondata.hdf5'
    # motionsdictpath = '../data/spbsl/motionsdic.pkl'
    worddictpath = '../data/spbsl/WordDict.pkl'
    subtitledictpath = '../data/spbsl/SubtitleDict.pkl'
    annotationdictpath = '../data/spbsl/annotationindex.hdf5'

    if testcode == 0:
        # calculate the shapelet info with all the data
        cls_shapelet = ShapeletsFinding(motionhdf5filepath, worddictpath, subtitledictpath)
        # consider: 500, thank:2153, supermarket:60, weekend: 76, expert: 99
        # cls_shapelet.train('supermarket', method=1)
        cls_shapelet.train(method=1)

    elif testcode == 1:
        with h5py.File(annotationdictpath, 'r') as f:
            words = list(f.keys())
        
        # EDrecordfile = '../data/spbsl/temprecord.hdf5'
        # Netrecordfile = '../data/spbsl/temprecordnet.hdf5'

        # if method == 2:
        #     recordfile = EDrecordfile
        # elif method == 1:
        #     recordfile = Netrecordfile
            
        cls_shapelet = ShapeletsFinding(motionhdf5filepath, worddictpath, subtitledictpath)
        cls_shapelet.train(words, method=method, overwrite=retrain)

        # PE.CalculateRecallRate_allh5file(annotationdictpath, recordfile)
        # PE.CalculateRecallRate(annotationdictpath, recordfile)
        
        
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testcode', type=int, default=1)
    parser.add_argument('-r', '--retrain', action='store_true')
    parser.add_argument('-m', '--method', type=int, default=2)
    args = parser.parse_args()
    testcode = args.testcode
    retrain = args.retrain
    method = args.method
    RunTest(testcode, method, retrain)