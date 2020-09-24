'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-09-24 16:27:46
'''
import utilmx
import numpy as np
from sklearn.preprocessing import scale


def Shapelets_Rangelength(pos_samples, neg_samples, lengthrange):
    '''
    description: 
    param {type} 
    return {type} 
    author: mario
    ''' 
    low, high = lengthrange
    for length in range(low, high+1):
        for index, pos_sample in enumerate(pos_samples):
            for q_index in range(pos_samples.shape[0]-length+1):
                query = pos_sample[q_index:q_index+length]


def ShapeletsScore(query, pos_sample, neg_samples):
    norm_query = scale(query)