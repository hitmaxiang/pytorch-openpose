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

from scipy import fft
from srcmx import utilmx
from scipy.signal import convolve2d, convolve
from sklearn.linear_model import LogisticRegression as LR


class SlidingDisNet(nn.Module):
    def __init__(self, shape, query=None):
        super().__init__()
        m, d = shape
        # 因为 query 最后要与 sequence 做卷积， 所以尺寸应该为 1 x D x m
        if query is not None:
            self.query = query
        else:
            self.query = torch.randn(1, d, m, dtype=torch.float32, requires_grad=True)
        
        self.weight = torch.ones(1, d, m)
        self.CLS = nn.Linear(1, 2)
        self.optimer = torch.optim.Adam([self.query, self.CLS.weight, self.CLS.bias], lr=1e-1)
        # self.optimer = torch.optim.SGD([self.query, self.CLS.weight, self.CLS.bias], lr=0.5)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def __call__(self, X):
        # X 的 shape 应该是 （Batch x D x T）
        with torch.no_grad():
            return self.slidingdistance(X)

    def forward(self, X, label):
        # X 的 shape 应该是 （Batch x D x T）
        DIS = self.slidingdistance(X)
        DIS = torch.min(DIS, dim=1, keepdim=True)[0]
        meadis = torch.mean(DIS[label == 1]) - torch.mean(DIS[label == 0])
        Y = self.CLS(DIS)
        return Y, meadis

    def train(self, X, Y, epochs=10000):
        # Batchsize = X.shape[0]
        temploss = float('inf')
        counter = 0
        for i in range(epochs):
            Y_e, meadis = self.forward(X, Y)
            loss = self.loss_fn(Y_e, Y) + meadis
            # if loss.item() > temploss:
            #     if counter > 100:
            #         break
            #     else:
            #         counter += 1
            # else:
            #     temploss = loss.item()
            #     counter = 0

            self.optimer.zero_grad()
            loss.backward()
            self.optimer.step()
            with torch.no_grad():
                _, predics = torch.max(Y_e, dim=1)
                correct = int((predics == Y).sum())
            if i % 100 == 0:
                print('epoch: %d, loss %f, accuracy: %f' % (i, loss, correct/Y.shape[0]))
        
        print('\n finish the train:\n')
        print('epoch: %d, loss %f, accuracy: %f' % (i, loss, correct/Y.shape[0]))
        # print('wieght: %f, bias:  %f' % (self.CLS.weight.item(), self.CLS.bias.item()))
    
    def localizeshape(self, X):
        with torch.no_grad():
            DIS = self.slidingdistance(X)
            dis, loc = torch.min(DIS, dim=1)
        
        return loc, dis
    
    def getshapelet(self):
        shapelet = self.query.detach().squeeze(0).permute(1, 0)
        return shapelet
    
    def slidingdistance(self, X):
        QQ = torch.sum(torch.square(self.query))
        XX = F.conv1d(torch.square(X), weight=self.weight)
        QX = F.conv1d(X, weight=self.query)
        DIS = QQ + XX - 2 * QX
        DIS = torch.sqrt(DIS.squeeze(dim=1))

        return DIS
    
    def softmin(self, x, alpha=-20):
        x1 = torch.exp(alpha*x)
        x2 = x * x1
        y = torch.sum(x2, dim=1)/torch.sum(x1, dim=1)
        return y


class ShapeletMatrix():
    def __init__(self):
        self.cls = LR()
    
    def train(self, X, labels, m_len):
        # X shape batchsize x D x T
        bestscore = 0
        bestloss = float('inf')

        N, D, T = X.shape
        catsamples = X.permute(0, 2, 1).view(-1, D)
        if torch.cuda.is_available():
            catsamples = catsamples.cuda()

        for i in range(N):
            Dis_sample = np.zeros((N, T-m_len+1))
            Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
            # time_0 = time.time()

            begin = T * i
            end = T * (i+1)
            with torch.no_grad():
                DISMAT = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
            # time_1 = time.time()
            DISMAT = DISMAT.cpu()
            for j in range(N):
                index_by = T * j
                index_ey = index_by + T - m_len + 1

                DisMat = DISMAT[:, index_by:index_ey]

                Dis_sample[j], Dis_loc[j] = torch.min(DisMat, dim=-1)
                # Dis_loc[j] = np.argmin(DisMat, axis=-1)
                # Dis_sample[j] = np.min(DisMat, axis=-1)
            
            # 针对每一个可能的 candidate sign, 求解它的score
            for candin_index in range(T-m_len+1):
                # score = self.Bi_class(Dis_sample[:, candin_index], labels)
                score = self.Bipartition_score(Dis_sample[:, candin_index], labels.numpy())
                # print(score, score1)

                loss = np.mean(Dis_sample[:, candin_index][labels == 1])
                if score > bestscore:
                    bestscore = score
                    shapeindex = i
                    bestloss = loss
                    dis = Dis_sample[:, candin_index]
                    locs = Dis_loc[:, candin_index]
                elif score == bestscore and loss < bestloss:
                    shapeindex = i
                    bestloss = loss
                    locs = Dis_loc[:, candin_index]
                    dis = Dis_sample[:, candin_index]
            # time_2 = time.time()
            # print('%f----%f' % (time_1-time_0, time_2-time_1))
            print('%d/%d--->loss: %f, accuracy: %f' % (i, N, bestloss, bestscore))

        self.shapeindex = shapeindex
        self.locs = locs
        self.dis = dis

    def Bi_class(self, dis, label):
        dis = dis[:, np.newaxis]
        self.cls.fit(dis, label)
        return self.cls.score(dis, label)
    
    def Bipartition_score(self, distances, labels):
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
        
        score = maxcorrect/len(distances)

        return score


def slidingdotproductfft(Q, T):
    # m, n = len(Q), len(T)
    
    # Q_ar = np.pad(np.flip(Q, axis=0), ((0, n-m), (0, 0)), mode='constant', constant_values=(0, 0))
    QT = fft.ifft(fft.fft(T, axis=0)*fft.fft(Q, axis=0), axis=0)
    QT = np.sum(QT, axis=-1)
    return np.real(QT)


def slidingdotproductconvl(Q, T):
    # Qr = np.flip(Q)
    products = convolve2d(T, Q, mode='valid')
    # products = convolve(T, Q, mode='valid')
    return products


def slidingdotproductorigi(Q, T):
    m, n = len(Q), len(T)
    scores = np.zeros((n-m+1))
    for i in range(len(scores)):
        scores[i] = np.sum(Q * T[i:m+i])
    return scores


def slidingdotproductconvltorch(Q, T):
    # Q = np.transpose(Q)[np.newaxis, :]
    # T = np.transpose(T)[np.newaxis, :]
    products = F.conv1d(T, Q)
    # products = products[0, 0, :].numpy()
    return products


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


def slidingDistance_convol(Q, T):
    m = len(Q)
    n = len(T)
    DIS = np.zeros((n-m+1,))
    # DIS += np.sum(np.square(Q))
    SS = np.sum(Q*Q)
    TT = np.sum(T*T, axis=-1)
    offset = TT[m:] - TT[:(n-m)]
    cumoffset = np.cumsum(offset)
    sum1 = np.sum(TT[:m])
    DIS[0] = sum1
    DIS[1:] = sum1 + cumoffset 
    # Q_r = np.flip(Q)
    # J_3 = slidingdotproductconvl(Q_r, T)[:, 0]
    # J_3 = slidingdotproductorigi(Q, T)
    
    Q = torch.from_numpy(np.transpose(Q)[np.newaxis, :])
    T = torch.from_numpy(np.transpose(T)[np.newaxis, :])
    J_3 = slidingdotproductconvltorch(Q, T)[0, 0, :].numpy()

    DIS += SS
    DIS -= 2*J_3
    return np.sqrt(DIS)


def matrixprofile_torch(sequenceA, sequenceB, m, DisMat):
    # l_1 = len(sequenceA)
    # l_2 = len(sequenceB)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    # DisMat = DisMat.to(sequenceA.device)
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
    return DisMat


def Test(testcode):
    if testcode == 0:
        # test the dotproduct 
        Q = np.random.rand(10, 3)
        T = np.random.rand(500, 3)
        Iters = 1000

        t0 = time.time()
        m, n = len(Q), len(T)
        Q_ar = np.pad(np.flip(Q, axis=0), ((0, n-m), (0, 0)), mode='constant', constant_values=(0, 0))
        for _ in range(Iters):
            d0 = slidingdotproductfft(Q_ar, T)[m-1:]

        t = time.time()
        for _ in range(Iters):
            d1 = slidingdotproductorigi(Q, T)
        t1 = time.time()
        
        Q = np.flip(Q)
        for _ in range(Iters):
            d2 = slidingdotproductconvl(Q, T)
            
        Q = np.flip(Q)
        t2 = time.time()

        Q = torch.from_numpy(np.transpose(Q)[np.newaxis, :])
        T = torch.from_numpy(np.transpose(T)[np.newaxis, :])
        for _ in range(Iters):
            d3 = slidingdotproductconvltorch(Q, T)
        d3 = d3[0, 0, :].numpy()
        t3 = time.time()

        print(np.allclose(d1, d0))
        print('Orig %0.3f ms, zero approach %0.3f ms' % ((t1 - t) * 1000., (t - t0) * 1000.))
        print('Speedup ', (t1 - t) / (t - t0))

        print(np.allclose(d1, d2[:, 0]))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))

        print(np.allclose(d1, d3))
        print('Orig %0.3f ms, third approach %0.3f ms' % ((t1 - t) * 1000., (t3 - t2) * 1000.))
        print('Speedup ', (t1 - t) / (t3 - t2))

    elif testcode == 1:
        Q = np.random.rand(10, 3)
        T = np.random.rand(500, 3)
        Iters = 10

        t0 = time.time()
        for _ in range(Iters):
            d0 = SlidingDistance(Q, T)
        t1 = time.time()
        for _ in range(Iters):
            d1 = slidingDistance_convol(Q, T)
        
        t2 = time.time()

        print(np.allclose(d1, d0))
        print('Orig %0.3f ms, zero approach %0.3f ms' % ((t1 - t0) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t0) / (t2 - t1))
    
    elif testcode == 2:
        # 测试 sliding distance net 相关
        m, d = 10, 3
        Q = np.random.rand(m, d)
        T = np.random.rand(2000, d)

        Iters = 1000

        t_0 = time.time()
        for _ in range(Iters):
            dis1 = SlidingDistance(Q, T)
        t_1 = time.time()

        Q_c = torch.from_numpy(Q).permute(1, 0).unsqueeze(0).to(dtype=torch.float32)
        T_c = torch.from_numpy(T).permute(1, 0).unsqueeze(0).to(dtype=torch.float32)

        Net = SlidingDisNet((m, d), Q_c)

        t_2 = time.time()
        for _ in range(Iters):
            # dis2 = Net(T_c).squeeze(0).numpy()
            dis2 = Net.forward(T_c).squeeze(0).numpy()
        t_3 = time.time()

        print(np.allclose(dis1, dis2))
        print('Orig %0.3f ms, zero approach %0.3f ms' % ((t_1 - t_0) * 1000., (t_3 - t_2) * 1000.))
        print('Speedup ', (t_1 - t_0) / (t_3 - t_2))
    
    elif testcode == 3:
        m, d = 20, 5
        samplenum = 500
        Ts = torch.randn(samplenum, d, 500)
        Y = torch.randint(low=0, high=2, size=(samplenum,))
        # Y = torch.scatter(torch.zeros(samplenum, 2), 1, Y, 1.0)

        # 修改一些
        real_m = 20
        location = torch.randint(500-real_m, size=(samplenum,))
        query = torch.randn(d, real_m)
        for i in range(samplenum):
            if Y[i] == 1 and np.random.rand() > 0.2:
                Ts[i, :, location[i]:location[i]+real_m] = query + 0.01 * torch.randn(d, real_m)
        
        query_default = query + torch.randn(d, real_m)
        query_default = query_default.unsqueeze(0)
        # Net = SlidingDisNet(shape=(m, d), query=query_default)
        # Net = SlidingDisNet(shape=(m, d))
        import sys
        sys.path.append('./srcmx')
        from srcmx import shapeletmodel
        Net = shapeletmodel.ShapeletNetModel(shape=(m, d), query=query_default)

        Net.train(Ts, Y)
        locs, dis = Net.localizeshape(Ts)
        shapelet = Net.getshapelet()

        plt.switch_backend('agg')

        plt.figure(0)
        for i in range(samplenum):
            if Y[i] == 1:
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

        # 使用 LR 分类器进行分类
        classer = LR()
        X = dis.unsqueeze(-1).numpy()
        Y = Y.numpy()
        classer.fit(X, Y)
        print(classer.score(X, Y))
    
    elif testcode == 4:
        # prepare the data
        m, d = 20, 1
        samplenum = 200
        Ts = torch.randn(samplenum, d, 500)
        Y = torch.randint(low=0, high=2, size=(samplenum,))
        # Y = torch.scatter(torch.zeros(samplenum, 2), 1, Y, 1.0)

        # 修改一些
        real_m = 20
        location = torch.randint(500-real_m, size=(samplenum,))
        query = torch.randn(d, real_m)
        for i in range(samplenum):
            if Y[i] == 1 and np.random.rand() > 0.2:
                Ts[i, :, location[i]:location[i]+real_m] = query + 0.01 * torch.randn(d, real_m)
        
        extracter = ShapeletMatrix()
        extracter.train(Ts, Y, m)

        locs = extracter.locs
        shapindex = extracter.shapeindex
        dis = extracter.dis
        shapelet = Ts[shapindex, 0, locs[shapindex]:locs[shapindex]+m]

        plt.switch_backend('agg')
        plt.figure(0)
        for i in range(samplenum):
            if Y[i] == 1:
                plt.plot(Ts[i, 0, locs[i]:locs[i]+m], 'b', linewidth=1)

        plt.plot(shapelet, 'r', linewidth=2)
        plt.plot(query[0], 'g--', linewidth=2)
        plt.title('positive')
        plt.savefig('pos.jpg')

        plt.figure(1)
        for i in range(samplenum):
            if Y[i] == 0:
                plt.plot(Ts[i, 0, locs[i]:locs[i]+m], 'b', linewidth=1)
        plt.plot(shapelet, 'r', linewidth=2)
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



            



        
        
if __name__ == "__main__":
    Test(3)





    

