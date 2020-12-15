from srcmx import utilmx
import torch
import numpy as np


def SlidingDistanceSquare_torch(pattern, sequence):
    m = len(pattern)
    n = len(sequence)
    _len = n - m + 1
    dist = torch.square(pattern[0] - sequence[:_len])
    for i in range(1, m):
        dist += torch.square(pattern[i] - sequence[i:i+_len])
    if len(dist.shape) == 2:
        dist = torch.sum(dist, axis=-1)
    return dist


def matrixprofile_torch_ori(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    DisMat = DisMat.to(sequenceA.device)
    # DisMat.zero_()
    for i in range(DisMat.shape[0]):
        DisMat[i] = utilmx.SlidingDistance_torch(sequenceA[i:i+m], sequenceB)
    
    return DisMat


def matrixprofile_torch_ori_1(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    DisMat = DisMat.to(sequenceA.device)
    # DisMat.zero_()
    for i in range(DisMat.shape[0]):
        DisMat[i] = SlidingDistanceSquare_torch(sequenceA[i:i+m], sequenceB)
    
    return torch.sqrt(DisMat)


def matrixprofile_torch(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    DisMat = DisMat.to(sequenceA.device)
    # if torch.cuda.is_available():
    #     DisMat = DisMat.cuda()
    DisMat[0, :] = utilmx.SlidingDistance_torch(sequenceA[:m], sequenceB)
    DisMat[:, 0] = utilmx.SlidingDistance_torch(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = torch.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= torch.square(sequenceA[r-1]-sequenceB[:-m])
        offset = torch.sum(offset, axis=-1)
        DisMat[r, 1:] = torch.sqrt(torch.square(DisMat[r-1, :-1])+offset)
    return DisMat


def matrixprofile_torch_1(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
    # DisMat = torch.zeros(l_1-m+1, l_2-m+1)
    DisMat = DisMat.to(sequenceA.device)
    # if torch.cuda.is_available():
    #     DisMat = DisMat.cuda()
    DisMat[0, :] = SlidingDistanceSquare_torch(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistanceSquare_torch(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = torch.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= torch.square(sequenceA[r-1]-sequenceB[:-m])
        offset = torch.sum(offset, axis=-1)
        # print(offset)
        # print(DisMat[r-1, :-1])
        DisMat[r, 1:] = DisMat[r-1, :-1]+offset
        # print(DisMat[r, 1:])
        # valid = torch.min(DisMat[r, 1:]).item() < 0
        # if valid:
        #     print(DisMat[r, 1:])
    # print(torch.min(DisMat))
    return torch.sqrt(DisMat)


def samples2tensor(samples):
    # samples 为样本序列，每个的格式为 T_i x D，具有不同的长度
    # 获取
    minvalue, maxvalue = 10000, 12000
    batchsize = len(samples)
    lengths = [len(x) for x in samples]
    T, D = max(lengths), samples[0].shape[1]

    # sample_tensor = torch.zeros(batchsize, D, T, dtype=torch.float32)
    # sample_tensor[:] = torch.randint(1000, 1100, size=)
    sample_tensor = torch.randint(minvalue, maxvalue, size=(batchsize, D, T)).float()+torch.rand(batchsize, D, T)
    for i in range(batchsize):
        sample_tensor[i, :, :lengths[i]] = torch.from_numpy(samples[i]).transpose(0, 1).float()
    
    return sample_tensor, lengths


if __name__ == "__main__":
    TEST = 2
    
    if TEST == 0:
        for i in range(100):
            A = torch.randn(200, 12)
            B = torch.randn(300, 12)
            m = 10

            dis1 = matrixprofile_torch_ori(A, B, m)
            dis2 = matrixprofile_torch_ori_1(A, B, m)
            print(i, torch.allclose(dis1, dis2))
    elif TEST == 1:
        for i in range(100):
            A = torch.randn(200, 12)
            # A = torch.randint(100, 120, size=(200, 12)).float()
            # A[:-20] = torch.randn(180, 12)
            B = torch.randn(300000, 12)
            # B[-15:] = torch.randint(1000, 1200, size=(15, 12)).float()
            m = 10

            dis1 = matrixprofile_torch_ori(A, B, m)
            # dis2 = matrixprofile_torch(A, B, m)
            dis2 = matrixprofile_torch_1(A, B, m)
            print(torch.max(dis1), torch.max(dis2))
            print(i, torch.allclose(dis1, dis2))
    elif TEST == 2:
        m, d = 3, 1
        minlen, maxlen = 500, 501
        samplenum = 2000
        for _ in range(100):
            samples = []
            for _ in range(samplenum):
                sample = np.random.randn(np.random.randint(minlen, maxlen), d)
                samples.append(sample)
            Ts, lengths = samples2tensor(samples)

            N, D, T = Ts.shape
            catsamples = torch.reshape(Ts.permute(0, 2, 1), (-1, D))
            for i in range(N):
                dis1 = matrixprofile_torch_ori(catsamples[T*i:T*(i+1)], catsamples, m)
                dis2 = matrixprofile_torch(catsamples[T*i:T*(i+1)], catsamples, m)
                # dis2 = matrixprofile_torch_ori_1(catsamples[T*i:T*(i+1)], catsamples, m)
                dis2[torch.isnan(dis2)] = float('inf')
                print(torch.max(dis1), torch.max(dis2))
                disvalid = torch.allclose(torch.min(dis1, dim=-1)[0], torch.min(dis2, dim=-1)[0])
                print(i, disvalid)
                print(i, torch.allclose(torch.min(dis1, dim=-1)[1], torch.min(dis2, dim=-1)[1]))
                print(i, torch.max(torch.abs(dis1-dis2)))
                if not disvalid:
                    print(torch.min(dis2, dim=-1)[0])

