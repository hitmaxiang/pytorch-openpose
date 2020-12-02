import numpy
import numpy as np
import time
import scipy.fft as fft
import torch


def sliding_dist(A, B):
    m = len(A)
    n = len(B)
    dist = numpy.zeros(n-m+1)
    for i in range(n-m+1):
        subrange = B[i:i+m]
        distance = numpy.linalg.norm(A-subrange)
        dist[i] = distance
    return dist


def SlidingDistance_torch(pattern, sequence):
    m = len(pattern)
    n = len(sequence)
    _len = n - m + 1
    dist = torch.square(pattern[0] - sequence[:_len])
    for i in range(1, m):
        dist += torch.square(pattern[i] - sequence[i:i+_len])
    if len(dist.shape) == 2:
        dist = torch.sum(dist, axis=-1)
    return torch.sqrt(dist)


def matrixprofile_torch(sequenceA, sequenceB, DisMat, m):
    # l_1 = len(sequenceA)
    # l_2 = len(sequenceB)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    # if torch.cuda.is_available():
    #     DisMat = DisMat.cuda()
    DisMat[0, :] = SlidingDistance_torch(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistance_torch(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = torch.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= torch.square(sequenceA[r-1]-sequenceB[:-m])
        offset = torch.sum(offset, axis=-1)
        DisMat[r, 1:] = torch.sqrt(DisMat[r-1, :-1]**2+offset)
    return DisMat


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


def matrixprofile(sequenceA, sequenceB, DisMat, m):
    # l_1 = len(sequenceA)
    # l_2 = len(sequenceB)
    # DisMat = np.zeros((l_1-m+1, l_2-m+1))
    DisMat[0, :] = SlidingDistance(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistance(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = np.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= np.square(sequenceA[r-1]-sequenceB[:-m])
        offset = np.sum(offset, axis=-1)
        DisMat[r, 1:] = np.sqrt(DisMat[r-1, :-1]**2+offset)
    return DisMat


def matrixprofile_origi(sequenceA, sequenceB, DisMat, m):
    for i in range(DisMat.shape[0]):
        DisMat[i] = SlidingDistance(sequenceA[i:m+i], sequenceB)
        # DisMat[i] = sliding_dist(sequenceA[i:m+i], sequenceB)
    return DisMat


def Test(testcode):
    if testcode == 0:
        A = numpy.random.rand(10, 3)
        B = numpy.random.rand(500, 3)
        x = 1000
        t = time.time()

        for _ in range(x):
            d1 = sliding_dist(A, B)
        t1 = time.time()

        for _ in range(x):
            d2 = SlidingDistance(A, B)
        t2 = time.time()

        print(numpy.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))
    elif testcode == 1:
        A = np.random.rand(600, 3)
        B = np.random.rand(500, 3)
        m = 10
        x = 1000

        t = time.time()
        l_1 = len(A)
        l_2 = len(B)
        DisMat = np.zeros((l_1-m+1, l_2-m+1))
        for _ in range(x):
            d1 = matrixprofile_origi(A, B, DisMat, m)
        t1 = time.time()

        for _ in range(x):
            d2 = matrixprofile(A, B, DisMat, m)
        t2 = time.time()

        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        DisMat = torch.from_numpy(DisMat)
        if torch.cuda.is_available():
            A = A.cuda()
            B = B.cuda()
            DisMat = DisMat.cuda()
        with torch.no_grad():
            for _ in range(x):
                d3 = matrixprofile_torch(A, B, DisMat, m)
        d3 = d3.cpu().numpy()
        t3 = time.time()


        print(numpy.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))
        print(numpy.allclose(d1, d3))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t3 - t2) * 1000.))
        print('Speedup ', (t1 - t) / (t3 - t2))
    elif testcode == 2:
        A = numpy.random.rand(20, 24)
        B = numpy.random.rand(500000, 24)
        x = 10
        t = time.time()

        for _ in range(x):
            d1 = SlidingDistanceFFT(A, B)
        t1 = time.time()

        for _ in range(x):
            d2 = SlidingDistance(A, B)
        t2 = time.time()

        print(numpy.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))
    
    elif testcode == 3:
        # prepare the data
        N = 10
        datasets = []
        for i in range(N):
            L = np.random.randint(250, 300)
            datasets.append(np.random.rand(L, 24))
        # datasequence = np.concatenate(datasets, axis=0)

        t_0 = time.time()
        # using the couple-2-couple methods
        for i in range(N):
            samples1 = datasets[i]
            for j in range(N):
                samples2 = datasets[j]
                datamat = matrixprofile(samples1, samples2, 10)
        t_1 = time.time()

        # using the concatenate style

        datasequence = np.concatenate(datasets, axis=0)
        datasequence = torch.from_numpy(datasequence)
        if torch.cuda.is_available():
            datasequence = datasequence.cuda()
        t_11 = time.time()
        with torch.no_grad():
            datamats = matrixprofile_torch(datasequence, datasequence, 10)
        t_2 = time.time()

        print('Orig %0.3f ms, second approach %0.3f ms' % ((t_1 - t_0) * 1000., (t_2 - t_11) * 1000.))
        print('Speedup ', (t_1 - t_0) / (t_2 - t_11))
    
    elif testcode == 4:
        A = numpy.random.rand(20, 24)
        B = numpy.random.rand(5000, 24)
        x = 1000
        t = time.time()

        for _ in range(x):
            d1 = SlidingDistance(A, B)
        t1 = time.time()
        t11 = time.time()

        A = torch.from_numpy(A)
        B = torch.from_numpy(B)
        if torch.cuda.is_available():
            A = A.cuda()
            B = B.cuda()
        # t11 = time.time()
        with torch.no_grad():
            for _ in range(x):
                d2 = SlidingDistance_torch(A, B)
        t2 = time.time()

        d2 = d2.cpu().numpy()
        print(numpy.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t11) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t11))
    
    elif testcode == 5:
        # prepare the data
        N = 10
        datasets = []
        for i in range(N):
            L = np.random.randint(250, 300)
            datasets.append(torch.randn(L, 24))
        # datasequence = np.concatenate(datasets, axis=0)

        
        # using the couple-2-couple methods
        if torch.cuda.is_available():
            datasets = [x.cuda() for x in datasets]

        t_0 = time.time()
        with torch.no_grad():
            for i in range(N):
                samples1 = datasets[i]
                for j in range(N):
                    samples2 = datasets[j]
                    datamat = matrixprofile_torch(samples1, samples2, 10)
        t_1 = time.time()

        # using the concatenate style

        datasequence = torch.cat(datasets, dim=0)
        with torch.no_grad():
            datamats = matrixprofile_torch(datasequence, datasequence, 10)
        t_2 = time.time()

        print('Orig %0.3f ms, second approach %0.3f ms' % ((t_1 - t_0) * 1000., (t_2 - t_1) * 1000.))
        print('Speedup ', (t_1 - t_0) / (t_2 - t_1))





if __name__ == "__main__":
    Test(1)
