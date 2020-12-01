import numpy
import numpy as np
import time
import scipy.fft as fft


def sliding_dist(A, B):
    m = len(A)
    n = len(B)
    dist = numpy.zeros(n-m+1)
    for i in range(n-m+1):
        subrange = B[i:i+m]
        distance = numpy.linalg.norm(A-subrange)
        dist[i] = distance
    return dist


def sd_2(A, B):
    m = len(A)
    dist = numpy.square(A[0] - B[:-m])
    for i in range(1, m):
        dist += numpy.square(A[i] - B[i:-m+i])
    return numpy.sqrt(np.sum(dist, axis=-1))


def SlidingDistanceFFT(query, sequence):
    m = len(query)
    n = len(sequence)
    d = sequence.shape[1]

    SS = np.sum(np.square(query))
    TT = np.sum(np.square(sequence), axis=-1)
    offset = TT[m:] - TT[:(n-m)]
    cumoffset = np.cumsum(offset)
    sum1 = np.sum(TT[:m])
    J2 = sum1 + cumoffset

    Q = np.zeros((2*n, d))
    T = np.zeros_like(Q)
    T[:n] = sequence
    Q[:m] = np.flip(query, axis=0)

    FT = fft.fft(T, axis=0)
    FQ = fft.fft(Q, axis=0)

    QT = abs(fft.ifft(FT*FQ, axis=0))
    QT = np.sum(np.square(QT), axis=-1)[(m-1):n]
    QT += SS

    QT[0] += sum1
    QT[1:] += J2
    return QT


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


def matrixprofile(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = np.zeros((l_1-m+1, l_2-m+1))
    DisMat[0, :] = SlidingDistance(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistance(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = np.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= np.square(sequenceA[r-1]-sequenceB[:-m])
        offset = np.sum(offset, axis=-1)
        DisMat[r, 1:] = np.sqrt(DisMat[r-1, :-1]**2+offset)
    return DisMat


def matrixprofile_origi(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = np.zeros((l_1-m+1, l_2-m+1))
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

        for _ in range(x):
            d1 = matrixprofile_origi(A, B, m)
        t1 = time.time()

        for _ in range(x):
            d2 = matrixprofile(A, B, m)
        t2 = time.time()

        print(numpy.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))
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
        datasequence = np.concatenate(datasets, axis=0)

        t_0 = time.time()
        # using the couple-2-couple methods
        for i in range(N):
            samples1 = datasets[i]
            for j in range(N):
                samples2 = datasets[j]
                datamat = matrixprofile(samples1, samples2, 10)
        t_1 = time.time()

        # using the concatenate style
        # datasequence = np.concatenate(datasets, axis=0)
        datamats = matrixprofile(datasequence, datasequence, 10)
        t_2 = time.time()

        print('Orig %0.3f ms, second approach %0.3f ms' % ((t_1 - t_0) * 1000., (t_2 - t_1) * 1000.))
        print('Speedup ', (t_1 - t_0) / (t_2 - t_1))
        





if __name__ == "__main__":
    Test(3)
