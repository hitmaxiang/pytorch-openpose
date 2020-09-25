'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-09-25 23:04:39
'''
import utilmx
import numpy as np
import time
from sklearn.preprocessing import scale
from tslearn import metrics


def Shapelets_Rangelength(pos_samples, neg_samples, lengthrange):
    '''
    description: 
    param {type} 
    return {type} 
    author: mario
    ''' 
    low_len, high_len = lengthrange
    for length in range(low, high+1):
        for index, pos_sample in enumerate(pos_samples):
            for q_index in range(pos_samples.shape[0]-length+1):
                query = pos_sample[q_index:q_index+length]
                samples = pos_samples[:q_index] + pos_samples[q_index+1:] + neg_samples
                labels = [1] * (len(pos_samples)-1) + [0] * len(neg_samples)




def ShapeletsScore(query, samples, labels):
    Scores = np.zeros((len(samples),))
    Locations = np.zeros(len(samples), len(query))
    for index, sample in enumerate(samples):
        locas, score = metrics.dtw_subsequence_path(query, sample)
        Scores[index] = score
        Locations[index] = 



def Distance_shapelet_timeserie(norm_query, sort_index, Timeseries, mode=0):
    m = len(norm_query)
    Lower_env, upper_env = metrics.lb_envelope(norm_query, radius=2)
    # temp_minimum = float('inf')
    # temp_locas = 0
    # using the giving method
    # time_cumsum = np.cumsum(Timeserie, axis=0)
    # time_cumsum_squre = np.cumsum(Timeserie**2, axis=0)

    # true dtw distance
    if mode == 0:
        begintime = time.time()
        for Timeserie in Timeseries:
            temp_minimum = float('inf')
            temp_locas = 0
            time_cumsum = np.cumsum(Timeserie, axis=0)
            time_cumsum_squre = np.cumsum(Timeserie**2, axis=0)
            for i in range(len(Timeserie)-m+1):
                if i == 0:
                    t_mean = time_cumsum[m-1]/m
                    t_std = np.sqrt(time_cumsum_squre[m-1]/m - t_mean**2)
                else:
                    t_mean = (time_cumsum[m+i-1] - time_cumsum[i-1])/m
                    t_std = np.sqrt((time_cumsum_squre[m+i-1] - time_cumsum_squre[i-1])/m - t_mean**2)
                t_norm_query = (Timeserie[i:i+m] - t_mean)/t_std
                dis = metrics.dtw(norm_query, t_norm_query, global_constraint='sakoe_chiba', sakoe_chiba_radius=2)
                if dis < temp_minimum:
                    temp_minimum = dis
                    temp_locas = i
        print(' mode %d consume %f seconds' % (mode, time.time()-begintime))
    
    elif mode == 1:
        begintime = time.time()
        for Timeserie in Timeseries:
            temp_minimum = float('inf')
            temp_locas = 0
            for i in range(len(Timeserie)-m+1):
                t_norm_query = scale(Timeserie[i:i+m])
                dis = metrics.dtw(norm_query, t_norm_query, global_constraint='sakoe_chiba', sakoe_chiba_radius=2)
                if dis < temp_minimum:
                    temp_minimum = dis
                    temp_locas = i
        print(' mode %d consume %f seconds' % (mode, time.time()-begintime))
    elif mode == 2:
        # LB 1 dis
        begintime = time.time()
        for Timeserie in Timeseries:
            temp_minimum = float('inf')
            temp_locas = 0
            time_cumsum = np.cumsum(Timeserie, axis=0)
            time_cumsum_squre = np.cumsum(Timeserie**2, axis=0)
            for i in range(len(Timeserie)-m+1):
                if i == 0:
                    t_mean = time_cumsum[m-1]/m
                    t_std = np.sqrt(time_cumsum_squre[m-1]/m - t_mean**2)
                else:
                    t_mean = (time_cumsum[m+i-1] - time_cumsum[i-1])/m
                    t_std = np.sqrt((time_cumsum_squre[m+i-1] - time_cumsum_squre[i-1])/m - t_mean**2)
                t_norm_query = (Timeserie[i:i+m] - t_mean)/t_std
                lb_dis = 0
                for n_dim in range(t_norm_query.shape[1]):
                    # s_t_query = [1, 1, 1, 1]
                    # s_upper = np.array([[1], [2], [3], [4]]) 
                    # s_lower = np.array([[0], [0], [0], [0]])
                    s_t_query = t_norm_query[:, n_dim]
                    s_upper = upper_env[:, n_dim].reshape(upper_env.shape[0], -1)
                    s_lower = Lower_env[:, n_dim].reshape(Lower_env.shape[0], -1)
                    lb_dis += (metrics.lb_keogh(ts_query=s_t_query, envelope_candidate=(s_lower, s_upper)))**2

                lb_dis = np.sqrt(lb_dis)
                if lb_dis < temp_minimum:
                    # dis = lb_dis
                    dis = metrics.dtw(norm_query, t_norm_query, global_constraint='sakoe_chiba', sakoe_chiba_radius=2)
                    if dis < temp_minimum:
                        temp_minimum = dis
                        temp_locas = i
        print(' mode %d consume %f seconds' % (mode, time.time()-begintime))
    return temp_minimum, temp_locas


def Test(testcode):
    if testcode == 0:
        query = np.random.randint(10, size=(15, 2))
        samples = [np.random.randint(10, size=(np.random.randint(800, 1000), 2)) for i in range(100)]
        norm_query = scale(query)
        Distance_shapelet_timeserie(norm_query, None, samples, mode=0)
        Distance_shapelet_timeserie(norm_query, None, samples, mode=1)
        Distance_shapelet_timeserie(norm_query, None, samples, mode=2)


if __name__ == "__main__":
    Test(0)