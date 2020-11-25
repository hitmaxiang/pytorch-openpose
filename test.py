import numpy
import numpy as np
import time


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
        from pyts.classification import LearningShapelets

        X = [[1, 2, 2, 1, 2, 3, 2], [0, 2, 0, 2, 0, 2, 3], [0, 1, 2, 2, 1, 2, 2]]
        y = [0, 1, 0]
        clf = LearningShapelets(random_state=42, tol=0.01)
        clf.fit(X, y)





if __name__ == "__main__":
    Test(2)
