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
from scipy.signal import convolve2d, convolve
from scipy import fft


class SlidingDisNet(nn.Module):
    def __init__(self, shape, query=None):
        super().__init__()
        m, D = shape
        
    
    def __call__(self, X):
        # X 的 shape 应该是 （Batch x D x T）
        

    def forward(self, X):
        pass


        



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
        
        
if __name__ == "__main__":
    Test(1)





    

