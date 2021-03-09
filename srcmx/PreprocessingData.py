'''
Description: the preprocessing module for the data and filter
Version: 2.0
Autor: mario
Date: 2020-09-24 16:34:31
LastEditors: mario
LastEditTime: 2021-03-09 20:33:50
'''
import numpy as np
from numba import jit
import sklearn.preprocessing as PP
from copy import deepcopy
from sklearn.preprocessing import scale


def MotionJointFeatures(motiondata, datamode=None, featuremode=0):
    '''
    description: remove the no data joint in the data
    param {type} 
    return {type} 
    author: mario
    '''
    # TxCxD 片段的长度，节点的个数，特征的维度（2或者3）
    T, C, D = motiondata.shape

    # 为那些无效值赋值
    motiondata = ProcessingTheNonValue(motiondata.reshape(T, -1)).reshape(T, C, D)

    # 如果没有直接给出数据模式的话，可以直接根据数据的维度来进行判断
    if datamode is None:
        if C == 18:
            datamode = 'pose'
        elif C == 60:
            datamode = 'posehand'
        else:
            raise Exception('please input correct datamode')
    
    # the body mode motiondata has 18 joints of the whole body
    if featuremode == 0 or featuremode == 1:
        # 采用最为原始的方式，只选取比较有代表的节点
        posedata = motiondata[:, :18, :2]
        posedata -= posedata[:, 1].reshape(T, 1, -1)
        # 只选择两个胳膊的节点作为有效的特征
        wantchannel = [2, 3, 4, 5, 6, 7]
        posedata = posedata[:, wantchannel]

        if datamode == 'posehand':
            # left hand
            lefthanddata = motiondata[:, 18:18+21, :2]
            lefthanddata -= lefthanddata[:, 0].reshape(T, 1, -1)
            # 然后舍弃原点的数据
            lefthanddata = lefthanddata[:, 1:]

            # righ hand
            righthanddata = motiondata[:, 18+21:, :2]
            righthanddata -= righthanddata[:, 0].reshape(T, 1, -1)
            # 然后舍弃原点的数据
            righthanddata = righthanddata[:, 1:]

            data = np.concatenate((posedata, lefthanddata, righthanddata), axis=1)
        elif datamode == 'pose':
            data = posedata

    if featuremode == 1:
        # 主要的操作和 featuremode=0 的时候基本是一致的， 附加要做的是尺度归一化
        # 归一化的参数是 heart 与 nose 之间的垂直距离
        scale = motiondata[:, 0, 1] - motiondata[:, 1, 1] + 1e-5
        scale = scale[:, np.newaxis, np.newaxis]
        data = data/scale

    return data
    

def ProcessingTheNonValue(datamat, mode=0, sigma=2):
    # 在有关节点位置的数据有些时候是空的，所以需要补上前后的值
    
    if mode == 0:
        # 无论如何，只补上前边的有效值, 如果开始就是无效值呢？
        for c in range(datamat.shape[1]):
            for r in range(1, datamat.shape[0]):
                if datamat[r, c] == 0:
                    datamat[r, c] = datamat[r-1, c]
    elif mode == 1:
        # 只在窗口 +-sigma 的范围内进行填充
        copymat = deepcopy(datamat)
        for c in range(datamat.shape[1]):
            for r in range(datamat.shape[0]):
                if datamat[r, c] == 0:
                    for s in range(1, sigma+1):
                        if r-s >= 0 and copymat[r-s, c] != 0:
                            datamat[r, c] = copymat[r-s, c]
                            break
                        elif r+s < datamat.shape[0] and copymat[r+s, c] != 0:
                            datamat[r, c] = copymat[r+s, c]
                            break

    return datamat


@jit
def Strip_motion_times_data(array):
    '''
    description: remove the all zeros data in both end
    param: 
        array:  the array-like data (n_times, ndims)
    return: array_like data with striped
    author: mario
    '''
    sum_array = np.sum(array, axis=-1)
    
    for i in range(len(sum_array)):
        begin_index = i
        if sum_array[i] != 0:
            break
    
    for j in range(len(sum_array)):
        end_index = len(sum_array) - j
        if sum_array[end_index-1] != 0:
            break
    
    return array[begin_index:end_index]


def NormlizeData(samples, mode=0):
    '''
    description: 对样本数据进行归一化处理
    param: 
        samples: 时间序列样本, 因为每个时间序列样本的长度并不固定, 所以是以 list
            的形式给出的, 每个时间序列的格式为(n_times, n_dims)
        mode: 不同形式的归一化方式
    return: 归一化之后的时间序列样本
    author: mario
    '''
    # 对每个时间序列样本进行 zero_normlized 格式化
    if mode == 0:
        for s_id in range(len(samples)):
            samples[s_id] = scale(samples[s_id], axis=0)
    elif mode == 1:
        for s_id in range(len(samples)):
            samples[s_id] = PP.minmax_scale(samples[s_id], axis=0)

    return samples


if __name__ == '__main__':
    testcode = 0
    if testcode == 0:
        a = [[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]
        a = np.array(a)
        print(a)

        print(ProcessingTheNonValue(a.T, mode=1).T)