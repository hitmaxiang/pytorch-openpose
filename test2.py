from srcmx import utilmx
import numpy as np
import torch
import time


Testcode = 3

N = 1000
pN = 50
m_len = 10
samples = []

for i in range(N):
    n = np.random.randint(200, 300)
    samples.append(np.random.rand(n, 24))

# 为torch函数准备样本
dataset = [x.astype(np.float32) for x in samples]
datasets = [torch.from_numpy(x) for x in dataset]
catsamples = torch.cat(datasets, dim=0)


if Testcode == 0:
    if torch.cuda.is_available():
        datasets = [x.cuda() for x in datasets]
    # 测试 float32 与 float64 精确度
    for i in range(N):
        for j in range(N):
            with torch.no_grad():
                Dismat_torch = utilmx.matrixprofile_torch(datasets[i], datasets[j], m_len)
            Dismat_torch = Dismat_torch.cpu().numpy()
            Dismat_np = utilmx.matrixprofile(samples[i], samples[j], m_len)
            match = np.allclose(Dismat_torch, Dismat_np)
            if match is False:
                raise
            print('i:%d, j:%d  ' % (i, j), end='\r')

elif Testcode == 1:
    # 测试 长序列的累积误差对于最终精度的影响
    if torch.cuda.is_available():
        catsamples = catsamples.cuda()

    lengths = [0] + [len(x) for x in samples] 
    cumlength = np.cumsum(lengths)

    # 对于序列长度的限制比较大，不可以非常大
    with torch.no_grad():
        DISMAT = utilmx.matrixprofile_torch(catsamples[:cumlength[pN]], catsamples, m_len)
    DISMAT = DISMAT.cpu().numpy()

    for i in range(pN):
        index_bx = cumlength[i]
        index_ex = index_bx + lengths[i+1] - m_len + 1
        for j in range(len(samples)):
            index_by = cumlength[j]
            index_ey = index_by + lengths[j+1] - m_len + 1

            Dismat_torch = DISMAT[index_bx:index_ex, index_by:index_ey]

            Dismat_np = utilmx.matrixprofile(samples[i], samples[j], m_len)
            match = np.allclose(Dismat_torch, Dismat_np)
            if match is False:
                print('\n error %f occurs at (%d, %d)' % (np.linalg.norm(Dismat_np-Dismat_torch), i, j))
            print('i:%d, j:%d  ' % (i, j), end='\r')

elif Testcode == 2:
    # 测试短途累积的影响
    # 测试 长序列的累积误差对于最终精度的影响
    if torch.cuda.is_available():
        catsamples = catsamples.cuda()

    lengths = [0] + [len(x) for x in samples] 
    cumlength = np.cumsum(lengths)

    for i in range(pN):
        begin = cumlength[i]
        end = begin + lengths[i+1]
        with torch.no_grad():
            DISMAT = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
        DISMAT = DISMAT.cpu().numpy()
        for j in range(len(samples)):
            index_by = cumlength[j]
            index_ey = index_by + lengths[j+1] - m_len + 1

            Dismat_torch = DISMAT[:, index_by:index_ey]

            Dismat_np = utilmx.matrixprofile(samples[i], samples[j], m_len)
            match = np.allclose(Dismat_torch, Dismat_np)
            if match is False:
                print('\n error %f occurs at (%d, %d)' % (np.linalg.norm(Dismat_np-Dismat_torch), i, j))
            print('i:%d, j:%d  ' % (i, j), end='\r')

elif Testcode == 3:
    # 测试短途与长途之间的时间差异
    print(catsamples.device)
    if torch.cuda.is_available():
        catsamples = catsamples.cuda()
        datasets = [x.cuda() for x in datasets]
    print(catsamples.device)
    lengths = [0] + [len(x) for x in samples] 
    cumlength = np.cumsum(lengths)

    # 长途的
    t0 = time.time()
    # with torch.no_grad():
    #     DISMAT = utilmx.matrixprofile_torch(catsamples[:cumlength[pN]], catsamples, m_len)
    # DISMAT = DISMAT.cpu().numpy()
    # for i in range(pN):
    #     index_bx = cumlength[i]
    #     index_ex = index_bx + lengths[i+1] - m_len + 1
    #     for j in range(len(samples)):
    #         index_by = cumlength[j]
    #         index_ey = index_by + lengths[j+1] - m_len + 1

    #         Dismat_torch = DISMAT[index_bx:index_ex, index_by:index_ey]
    t1 = time.time()

    # 短途的
    for i in range(pN):
        begin = cumlength[i]
        end = begin + lengths[i+1]
        with torch.no_grad():
            DISMAT = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
        DISMAT = DISMAT.cpu().numpy()
        for j in range(len(samples)):
            index_by = cumlength[j]
            index_ey = index_by + lengths[j+1] - m_len + 1

            Dismat_torch = DISMAT[:, index_by:index_ey]

    t2 = time.time()
    print(t2-t1)
    # for i in range(pN):
    #     for j in range(N):
    #         with torch.no_grad():
    #             Dismat_torch = utilmx.matrixprofile_torch(datasets[i], datasets[j], m_len)
    #         Dismat_torch = Dismat_torch.cpu().numpy()

    t3 = time.time()
    for i in range(pN):
        for j in range(N):
            Dismat_np = utilmx.matrixprofile(samples[i], samples[j], m_len)
    
    t4 = time.time()
    origiterm = t4 - t3
    longterm = t1 - t0 
    shortterm = t2 - t1
    torchterm = t3 - t2
    print('long %f, short %f, torch %f, origi %f' % (longterm, shortterm, torchterm, origiterm))
    print(origiterm/np.array([longterm, shortterm, torchterm]))






