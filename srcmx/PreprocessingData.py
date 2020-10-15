'''
Description: the preprocessing module for the data and filter
Version: 2.0
Autor: mario
Date: 2020-09-24 16:34:31
LastEditors: mario
LastEditTime: 2020-10-15 20:24:41
'''
import numpy as np
from numba import jit
from sklearn.preprocessing import scale


def MotionJointSelect(motiondata, datamode, featuremode):
    '''
    description: remove the no data joint in the data
    param {type} 
    return {type} 
    author: mario
    '''
    
    # the body mode motiondata has 18 joints of the whole body
    if datamode == 'body':
        jointindex = [i for i in range(18)]
        # only the upper body, is needed
        if featuremode == 0:
            # remove the lower-limbs joints
            lower_limbs_joints = [8, 9, 10, 11, 12, 13]
            for i in lower_limbs_joints:
                jointindex.remove(i)
        elif featuremode == 1:
            # only upper limbs joints is provided
            jointindex = jointindex[:8]
    
        return motiondata[:, jointindex, :]


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

    return samples