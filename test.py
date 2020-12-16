'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-11-27 15:51:11
LastEditors: mario
LastEditTime: 2020-12-09 22:17:06
'''
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
    # 获取
    maxvalue = 12345
    batchsize = len(samples)
    lengths = [len(x) for x in samples]
    T, D = max(lengths), samples[0].shape[1]

    # sample_tensor = torch.zeros(batchsize, D, T, dtype=torch.float32)
    # sample_tensor[:] = torch.randint(1000, 1100, size=)
    sample_tensor = torch.randint(20000, 30000, size=(batchsize, D, T)).float()+torch.rand(batchsize, D, T)
    for i in range(batchsize):
        sample_tensor[i, :, :lengths[i]] = torch.from_numpy(samples[i]).transpose(0, 1).float()
    
    return sample_tensor, lengths


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
        
        extracter = ShapeletMatrix()
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

    elif testcode == 1:

        a = torch.randn(8, 1)
        # a[-4:] = maxvalue
        b = torch.randn(5, 1)
        # b[-3:] = maxvalue
        # b[5] = maxvalue
        m = 3
        l1 = len(a) - m + 1
        l2 = len(b) - m + 1
        dismat = torch.zeros(l1, l2)
        c = matrixprofile_torch(a, b, m, dismat)
        d = utilmx.matrixprofile_torch(a, b, m)
        print(torch.allclose(c, d))
        # print(a)
        # print(b)
        # print(c)
        # print(d)

        # print(c-d)
        # print(d)
        # print('the minimum is')
        # print(torch.min(c, dim=-1))
        # print(torch.min(d, dim=-1))


if __name__ == "__main__":
    Test(0)