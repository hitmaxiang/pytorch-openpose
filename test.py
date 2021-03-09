'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-11-27 15:51:11
LastEditors: mario
LastEditTime: 2021-03-09 21:50:09
'''
import sys
sys.path.append('./srcmx')

import h5py
import scipy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy import fft
from srcmx import utilmx
from srcmx import shapeletmodel
from srcmx import PreprocessingData as PD
from scipy.signal import convolve2d, convolve
from sklearn.linear_model import LogisticRegression as LR


def SlidingDistance(pattern, sequence):
    '''
    calculate the distance between pattern with all the candidate patterns in sequence
    the pattern has the shape of (m, d), and sequence has the shape of (n, d). the d is
    the dimention of the time series date.
    '''
    m = len(pattern)
    n = len(sequence)
    _len = n - m + 1
    dist = np.square(pattern[0] - sequence[:_len])
    dist = dist.astype(np.float32)
    for i in range(1, m):
        dist += np.square(pattern[i] - sequence[i:i+_len])
    if len(dist.shape) == 2:
        dist = np.sum(dist, axis=-1)
    return np.sqrt(dist)


def matrixprofile_torch(sequenceA, sequenceB, m, DisMat):
    # l_1 = len(sequenceA)
    # l_2 = len(sequenceB)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    # DisMat = DisMat.to(sequenceA.device)
    # sequenceA = sequenceA.double()
    # sequenceB = sequenceB.double()
    # DisMat = DisMat.double()
    DisMat.zero_()
    # if torch.cuda.is_available():
    #     DisMat = DisMat.cuda()
    DisMat[0, :] = utilmx.SlidingDistance_torch(sequenceA[:m], sequenceB)
    DisMat[:, 0] = utilmx.SlidingDistance_torch(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = torch.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= torch.square(sequenceA[r-1]-sequenceB[:-m])
        offset = torch.sum(offset, axis=-1)
        DisMat[r, 1:] = torch.sqrt(DisMat[r-1, :-1]**2+offset)
        # DisMat[r, :] = utilmx.SlidingDistance_torch(sequenceA[r:r+m], sequenceB)
    return DisMat.float()


def matrixprofile_torch_ori(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    # DisMat = DisMat.to(sequenceA.device)
    # DisMat.zero_()
    # if torch.cuda.is_available():
    #     DisMat = DisMat.cuda()
    for r in range(DisMat.shape[0]):
        DisMat[r, :] = utilmx.SlidingDistance_torch(sequenceA[r:r+m], sequenceB)
    return DisMat


class ShapeletMatrix():
    def __init__(self):
        self.cls = LR()
    
    def train(self, X, labels, m_len):
        # X shape batchsize x D x T 或者是 list T x D
        if isinstance(X, list):
            samples = [torch.from_numpy(x).float() for x in X]
            lengths = [len(x) for x in X]
            catsamples = torch.cat(samples, dim=0)
            N, T = len(lengths), max(lengths)
        else:
            N, D, T = X.shape
            catsamples = torch.reshape(X.permute(0, 2, 1), (-1, D))
            lengths = [T] * N
        
        bestscore = 0
        bestloss = float('inf')
        cumlength = np.cumsum([0] + lengths)
        shrink = - m_len + 1
        offset = [x + shrink for x in lengths]

        # 为了避免每次都要重新构建数据库，所以这里提前按照最大的sample构建距离矩阵
        DISMAT_pre = torch.zeros(T-m_len+1, cumlength[-1]+shrink, dtype=torch.float32)

        # 同样也会准备存放处理结果数据的
        MinDis = torch.zeros(N, T-m_len+1, dtype=torch.float32)
        MinLoc = torch.zeros(N, T-m_len+1, dtype=torch.int16)
        # tempdis = torch.zeros(T-m_len+1, T-m_len+1)
        # MinDis1 = torch.zeros(N, T-m_len+1, dtype=torch.float32)
        # MinLoc1 = torch.zeros(N, T-m_len+1, dtype=torch.int16)

        # MinDis2 = torch.zeros(N, T-m_len+1, dtype=torch.float32)
        # MinLoc2 = torch.zeros(N, T-m_len+1, dtype=torch.int16)

        if torch.cuda.is_available():
            catsamples = catsamples.cuda()
            DISMAT_pre = DISMAT_pre.cuda()
            MinDis = MinDis.cuda()
            MinLoc = MinLoc.cuda()

            # MinDis1 = MinDis1.cuda()
            # MinLoc1 = MinLoc1.cuda()

            # MinDis2 = MinDis2.cuda()
            # MinLoc2 = MinLoc2.cuda()
            # tempdis = tempdis.cuda()

        Dis_sample = np.zeros((N, T-m_len+1))
        Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
        dis = np.zeros((N,))
        locs = np.zeros((N,))
        
        for i in range(N):
            if labels[i] == 0:
                continue
            # Dis_sample = np.zeros((N, T-m_len+1))
            # Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)

            time_0 = time.time()
            begin, end = cumlength[i:i+2]
            # end = T * (i+1)
            with torch.no_grad():
                DISMAT_pre[:offset[i]] = matrixprofile_torch(
                    catsamples[begin:end], 
                    catsamples, 
                    m_len, 
                    DISMAT_pre[:offset[i]])
                
                # DISMAT = matrixprofile_torch_ori(
                #     catsamples[begin:end], 
                #     catsamples, 
                #     m_len)

                # DISMAT_1 = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
                time_1 = time.time()

                for j in range(N):
                    b_loc = cumlength[j]
                    e_loc = cumlength[j+1] + shrink
                    tempdis = DISMAT_pre[:offset[i], b_loc:e_loc]
                    MinDis[j, :offset[i]], MinLoc[j, :offset[i]] = torch.min(tempdis, dim=-1)
                    # MinDis1[j, :offset[i]], MinLoc1[j, :offset[i]] = torch.min(DISMAT[:, b_loc:e_loc], dim=-1)
                    # MinDis2[j, :offset[i]], MinLoc2[j, :offset[i]] = torch.min(DISMAT_1[:, b_loc:e_loc], dim=-1)

                # for k in range(N):
                #     b_loc = cumlength[k]
                #     e_loc = cumlength[k+1] + shrink
                #     MinDis2[k, :offset[i]], MinLoc2[k, :offset[i]] = torch.min(DISMAT_1[:, b_loc:e_loc], dim=-1)
                #     # tempdis = DISMAT_1[:, b_loc:e_loc]
                #     # MinDis2[j, :offset[i]], MinLoc2[j, :offset[i]] = torch.min(tempdis, dim=-1)

                # DISMAR_t = DISMAT_o.view(-1, N, T).permute(1, 0, 2)[:, :, :offset]
                # MinDis, MinLoc = torch.min(DISMAR_t, dim=-1)
                # print(torch.max(MinDis))

                Dis_sample = MinDis.cpu().numpy()
                Dis_loc = MinLoc.cpu().numpy()

                # Dis_sample1 = MinDis1.cpu().numpy()
                # Dis_loc1 = MinLoc1.cpu().numpy()

                # Dis_sample2 = MinDis2.cpu().numpy()
                # Dis_loc2 = MinLoc2.cpu().numpy()

                # # test the 
                # DISMAT_1 = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
                # # DISMAT_1[torch.isnan(DISMAT_1)] = float('inf')
                # for k in range(N):
                #     b_loc = cumlength[k]
                #     e_loc = cumlength[k+1] + shrink
                #     MinDis2[k, :offset[i]], MinLoc2[k, :offset[i]] = torch.min(DISMAT_1[:, b_loc:e_loc], dim=-1)
                #     # tempdis = DISMAT_1[:, b_loc:e_loc]
                #     # MinDis2[j, :offset[i]], MinLoc2[j, :offset[i]] = torch.min(tempdis, dim=-1)
                # # print(torch.max(MinDis))

                # Dis_sample2 = MinDis2.cpu().numpy()
                # Dis_loc2 = MinLoc2.cpu().numpy()

            
            # # 针对每一个可能的 candidate sign, 求解它的score
            # valid1 = np.allclose(Dis_sample2[:, :offset[i]], Dis_sample1[:, :offset[i]])
            # valid2 = np.allclose(Dis_loc2[:, :offset[i]], Dis_loc1[:, :offset[i]])
            # valid1 = np.allclose(Dis_sample[:, :offset[i]], Dis_sample1[:, :offset[i]])
            # valid2 = np.allclose(Dis_loc[:, :offset[i]], Dis_loc1[:, :offset[i]])
            # valid5 = np.allclose(Dis_sample[:, :offset[i]], Dis_sample2[:, :offset[i]])
            # valid6 = np.allclose(Dis_loc[:, :offset[i]], Dis_loc2[:, :offset[i]])
            # print(valid1, valid2, valid3, valid4, valid5, valid6)
            # print(valid1, valid2)
            # if not (valid1 and valid2):
            #     # print(valid1, valid2)
            #     dismatch = Dis_loc[:, :offset[i]] != Dis_loc1[:, :offset[i]]
            #     print('points nums: %d' % dismatch.sum())
            #     points = []
            #     for r in range(dismatch.shape[0]):
            #         for c in range(dismatch.shape[1]):
            #             if dismatch[r, c] == True:
            #                 points.append((r, c))
            #     print('labels:', end=' ')
            #     for item in points:
            #         print(labels[item[0]].item(), end=' ')
            #     print('\n maxabsdiff', end=' ')
            #     print(np.max(np.abs(Dis_sample[:, :offset[i]]-Dis_sample1[:, :offset[i]])))
            #     print('have nan?', np.isnan(Dis_sample[:, :offset[i]]).sum())
            # print(np.max(np.abs(Dis_sample[:, :offset[i]] - Dis_sample_2[:, :offset[i]])))

            for candin_index in range(lengths[i]-m_len+1):
                # print(np.max(np.abs(Dis_sample[:, candin_index]-Dis_sample_2[:, candin_index])))
                # score = self.Bi_class(Dis_sample[:, candin_index], labels)
                
                score = self.Bipartition_score(Dis_sample[:, candin_index], labels.numpy(), bestscore)
    
                # print(score, score1)

                loss = np.mean(Dis_sample[:, candin_index][labels == 1])
                if score > bestscore:
                    bestscore = score
                    shapeindex = i
                    bestloss = loss
                    dis = deepcopy(Dis_sample[:, candin_index])
                    locs = deepcopy(Dis_loc[:, candin_index])
                elif score == bestscore and loss < bestloss:
                    shapeindex = i
                    bestloss = loss
                    dis = deepcopy(Dis_sample[:, candin_index])
                    locs = deepcopy(Dis_loc[:, candin_index])
            time_2 = time.time()
            print('%f----%f' % (time_1-time_0, time_2-time_1))
            print('%d/%d--->loss: %f, accuracy: %f' % (i, N, bestloss, bestscore))

        self.shapeindex = shapeindex
        self.locs = locs
        self.dis = dis

    def Bi_class(self, dis, label):
        dis = dis[:, np.newaxis]
        self.cls.fit(dis, label)
        return self.cls.score(dis, label)
    
    def Bipartition_score(self, distances, labels, bestscore):
        '''
        description: 针对一个 distances 的 二分类的最大分类精度
        param: 
            pos_num: 其中 distance 的前 pos_num 个 的标签为 1, 其余都为 0
        return: 最高的分类精度, 以及对应的分割位置
        author: mario
        '''     
        dis_sort_index = np.argsort(distances)
        correct = len(distances) - labels.sum()
        Bound_correct = len(distances)
        maxcorrect = correct

        for i, index in enumerate(dis_sort_index):
            if labels[index] == 1:  # 分对的
                correct += 1
                if correct > maxcorrect:
                    maxcorrect = correct
            else:
                correct -= 1
                Bound_correct -= 1
            if correct == Bound_correct:
                break
            if (Bound_correct/len(distances)) < bestscore:
                break
        
        score = maxcorrect/len(distances)

        return score


def samples2tensor(samples):
    # samples 为样本序列，每个的格式为 T_i x D，具有不同的长度
    batchsize = len(samples)
    lengths = [len(x) for x in samples]
    T, D = max(lengths), samples[0].shape[1]

    sample_tensor = torch.zeros(batchsize, D, T, dtype=torch.float32)
    # sample_tensor[:] = float('inf')
    for i in range(batchsize):
        sample_tensor[i, :, :lengths[i]] = torch.from_numpy(samples[i]).transpose(0, 1).float()
    return sample_tensor


def TestNoneValueprocessImpact(motionfilepath):
    
    with h5py.File(motionfilepath, 'r') as motionfile:
        vieokeys = list(motionfile['posedata/pose'].keys())
        videokey = np.random.choice(vieokeys)
        N = len(motionfile['posedata/pose/%s' % videokey][:])
        m, T = 15, 300
        loc1 = np.random.randint(0, N-m)
        loc2 = np.random.randint(0, N-T)

        # construct the shaplet data
        svkey = '041'
        loc1 = 87720 + 229
        posedata = motionfile['posedata/pose/%s' % svkey][loc1:loc1+m].astype(np.float32)
        handdata = motionfile['handdata/hand/%s' % svkey][loc1:loc1+m].astype(np.float32)
        shapeletdata = np.concatenate((posedata, handdata), axis=1)
        shapeletdata = PD.MotionJointFeatures(shapeletdata, 'posehand', 1)
        shapeletdata = np.reshape(shapeletdata, (shapeletdata.shape[0], -1))

        # construct the sequence data
        posedata = motionfile['posedata/pose/%s' % videokey][:].astype(np.float32)
        handdata = motionfile['handdata/hand/%s' % videokey][:].astype(np.float32)

        seqdata1 = np.concatenate((posedata[loc2:loc2+T], handdata[loc2:loc2+T]), axis=1)
        seqdata1 = PD.MotionJointFeatures(seqdata1, 'posehand', 1)
        seqdata1 = np.reshape(seqdata1, (seqdata1.shape[0], -1))
        
        seqdata2 = np.concatenate((posedata, handdata), axis=1)
        seqdata2 = PD.MotionJointFeatures(seqdata2, 'posehand', 1)
        seqdata2 = np.reshape(seqdata2, (seqdata2.shape[0], -1))

        seqdata3 = seqdata2[loc2:loc2+T]

        dist1 = utilmx.SlidingDistance(shapeletdata, seqdata1)
        # dist2 = utilmx.SlidingDistance(shapeletdata, seqdata2)[loc2:loc2+T-m+1]
        dist3 = utilmx.SlidingDistance(shapeletdata, seqdata3)

        # print(np.allclose(dist1, dist2))
        if np.allclose(dist1, dist3) is False:
            print(abs(min(dist1) - min(dist3)))
        # print(np.allclose(dist1, dist3))
        # print(np.allclose(dist3, dist2))


def TestNoneValueprocessImpact2(motionfilepath):
    # the infomation of the shapelet 
    videokey = '041'
    begidx, endidx = 88720, 87974
    loc = 229

    with h5py.File(motionfilepath, 'r') as motionfile:
        # compare the difference of shapeletdata with two construction methods
        vieokeys = list(motionfile['posedata/pose'].keys())
        videokey = np.random.choice(vieokeys)
        N = len(motionfile['posedata/pose/%s' % videokey][:])
        m = np.random.randint(15, 40)
        T = np.random.randint(150, 300)

        begidx = np.random.randint(0, N-T+1)
        endidx = begidx + T
        loc = np.random.randint(0, T-m+1)

        # videokey = '041'
        # begidx = 87720
        # endidx = 87974
        # loc = 229
        # m = 15
        videokey = '073'
        begidx = 219537
        endidx = 219721
        loc = 24
        m = 15
                
        # construct the shaplet data
        posedata = motionfile['posedata/pose/%s' % videokey][begidx:endidx].astype(np.float32)
        handdata = motionfile['handdata/hand/%s' % videokey][begidx:endidx].astype(np.float32)
        segdata = np.concatenate((posedata, handdata), axis=1)
        shapeletdata = segdata[loc:loc+m]

        shapeletdata = PD.MotionJointFeatures(shapeletdata, 'posehand', 1)
        shapeletdata = np.reshape(shapeletdata, (shapeletdata.shape[0], -1))
        
        segdata = PD.MotionJointFeatures(segdata, 'posehand', 1)
        segdata = np.reshape(segdata, (segdata.shape[0], -1))
        shapeletdata1 = segdata[loc:loc+m]
        # diff = shapeletdata - shapeletdata1
        print(np.mean(abs(shapeletdata - shapeletdata1)))


def ListKeysofHdf5File(filepath):
    with h5py.File(filepath, 'r') as h5File:
        for word in h5File.keys():
            print(word)
            for keys in h5File[word].keys():
                print('\t%s' % keys)

def Test(testcode):
    if testcode == 0:
        # prepare the data
        real_m = 10
        m, d = 10, 24
        minlen, maxlen = 480, 500
        samplenum = 500
        samples = []
        for _ in range(samplenum):
            sample = np.random.randn(np.random.randint(minlen, maxlen), d)
            samples.append(sample)
        # Ts, lengths = samples2tensor(samples)
        # Ts = torch.randn(samplenum, d, 500)
        Y = torch.randint(low=0, high=2, size=(samplenum,))
        # Y = torch.scatter(torch.zeros(samplenum, 2), 1, Y, 1.0)
        lengths = [len(x) for x in samples]
        # 修改一些
        
        validindex = []
        query = torch.randn(real_m, d).numpy()
        for i in range(samplenum):
            if Y[i] == 1 and np.random.rand() > 0.5:
                location = torch.randint(lengths[i] - real_m, size=(samplenum,))
                validindex.append(i)
                samples[i][location[i]:location[i]+real_m] = query + 0.1 * torch.randn(real_m, d).numpy()
        
        # extracter = ShapeletMatrix()
        extracter = shapeletmodel.ShapeletMatrixModel()
        extracter.train(samples, Y, m)

        locs = extracter.locs
        shapindex = extracter.shapeindex
        dis = extracter.dis
        shapelet = samples[shapindex][locs[shapindex]:locs[shapindex]+m, 0]

        plt.switch_backend('agg')
        plt.figure(0)
        for i in range(samplenum):
            if Y[i] == 1 and i in validindex:
                plt.plot(samples[i][locs[i]:locs[i]+m, 0], 'b', linewidth=1)

        plt.plot(shapelet, 'r', linewidth=2)
        plt.plot(query[:, 0], 'g--', linewidth=2)
        plt.title('positive')
        plt.savefig('pos-1.jpg')

        plt.figure(1)
        for i in range(samplenum):
            if Y[i] == 0:
                plt.plot(samples[i][locs[i]:locs[i]+m, 0], 'b', linewidth=1)
        plt.plot(shapelet, 'r', linewidth=2)
        plt.plot(query[:, 0], 'g--', linewidth=2)
        plt.title('negative')
        plt.savefig('neg-1.jpg')

        plt.figure(2)
        for i in range(samplenum):
            if Y[i] == 1:
                plt.scatter([i], [dis[i].item()], c='r', marker='o')
            else:
                plt.scatter([i], [dis[i].item()], c='g', marker='*')
        plt.savefig('dis-1.jpg') 

    if testcode == 1:
        m, d = 20, 24
        samplenum = 500
        Ts = torch.randn(samplenum, d, 500)
        Y = torch.randint(low=0, high=2, size=(samplenum,))
        # Y = torch.scatter(torch.zeros(samplenum, 2), 1, Y, 1.0)
        realength = torch.randint(400, 500, size=(samplenum,))
        # 修改一些
        real_m = 20
        location = torch.randint(400-real_m, size=(samplenum,))
        query = torch.randn(d, real_m)
        validindex = []
        for i in range(samplenum):
            Ts[i, :, realength[i]:] = 0
            if Y[i] == 1 and np.random.rand() > 0.5:
                validindex.append(i)
                Ts[i, :, location[i]:location[i]+real_m] = query + 0.1 * torch.randn(d, real_m)
        
        query_default = query + torch.randn(d, real_m)
        query_default = query_default.unsqueeze(0)
        
        Net = shapeletmodel.ShapeletNetModel(shape=(m, d), query=query_default)
        # Net = shapeletmodel.ShapeletNetModel(shape=(m, d))

        Net.train(Ts, Y)
        locs, dis = Net.localizeshape(Ts)
        shapelet = Net.getshapelet()

        plt.switch_backend('agg')

        plt.figure(0)
        for i in range(samplenum):
            if Y[i] == 1 and (i in validindex):
                plt.plot(Ts[i, 0, locs[i]:locs[i]+m], 'b', linewidth=1)

        plt.plot(shapelet[:, 0], 'r', linewidth=2)
        plt.plot(query[0], 'g--', linewidth=2)
        plt.title('positive')
        plt.savefig('pos.jpg')

        plt.figure(1)
        for i in range(samplenum):
            if Y[i] == 0:
                plt.plot(Ts[i, 0, locs[i]:locs[i]+m], 'b', linewidth=1)
        plt.plot(shapelet[:, 0], 'r', linewidth=2)
        plt.plot(query[0], 'g--', linewidth=2)
        plt.title('negative')
        plt.savefig('neg.jpg')

        plt.figure(2)
        for i in range(samplenum):
            if Y[i] == 1:
                plt.scatter([i], [dis[i].item()], c='r', marker='o')
            else:
                plt.scatter([i], [dis[i].item()], c='g', marker='*')
        plt.savefig('dis.jpg')

    if testcode == 2:
        for i in range(1):
            # TestNoneValueprocessImpact('./data/spbsl/motiondata.hdf5')
            # print()
            TestNoneValueprocessImpact2('./data/spbsl/motiondata.hdf5')
    if testcode == 3:
        filepath = './data/spbsl/bk_shapeletED.hdf5'
        filepath2 = './data/spbsl/bk_shapeletNetED.hdf5'
        ListKeysofHdf5File(filepath)



if __name__ == "__main__":
    Test(3)