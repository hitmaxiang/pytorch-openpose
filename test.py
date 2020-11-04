'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-11-03 17:48:10
LastEditors: mario
LastEditTime: 2020-11-04 16:19:14
'''
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import scipy
from scipy.ndimage.filters import gaussian_filter
from copy import deepcopy
from numba import jit
import torch
import torch.nn.functional as F


def FindPeaks_2d(data, thre):
    # assert len(data.shape) == 2
    map_left = np.zeros(data.shape)
    map_left[1:, :] = data[:-1, :]
    map_right = np.zeros(data.shape)
    map_right[:-1, :] = data[1:, :]
    map_up = np.zeros(data.shape)
    map_up[:, 1:] = data[:, :-1]
    map_down = np.zeros(data.shape)
    map_down[:, :-1] = data[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (data > thre, data >= map_left, data >= map_right, data >= map_up, data >= map_down))
    cordi = np.nonzero(peaks_binary)
    # peaks_index = list(zip(z[0], z[1], z[2]))  # note reverse 
    cordi = np.array(cordi).T
    if cordi.shape[-1] == 3:
        records = [[] for i in range(data.shape[-1])]
        for y, x, c in cordi:
            records[c].append((x, y, data[y, x, c]))
        return records
    return cordi


def findpeaks_torch(data, thre):
    # data = torch.from_numpy(data)
    # if torch.cuda.is_available():
    #     data = data.cuda()
    
    with torch.no_grad():
        peaks_binary = data > thre
        torch.logical_and(peaks_binary, data >= F.pad(data, (1, 0))[:, :, :, :-1], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 1))[:, :, :, 1:], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 1, 0))[:, :, :-1, :], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 0, 1))[:, :, 1:, :], out=peaks_binary)
        peaks_binary = torch.nonzero(peaks_binary, as_tuple=False)
        # peaks_binary = peaks_binary.cpu().numpy()
    
    return peaks_binary



# @jit
def centroidnp(data):
    h, w = data.shape
    x = np.arange(w)
    y = np.arange(h)
    vx = data.sum(axis=0)
    vx /= vx.sum()
    vy = data.sum(axis=1)
    vy /= vy.sum()    
    return int(np.dot(vx, x)), int(np.dot(vy, y))


# @jit
def peaklables(data, thre):
    data[data < thre] = 0
    labels, nums = scipy.ndimage.label(data)
    peak_slices = scipy.ndimage.find_objects(labels)
    centroids = []
    for peak_slice in peak_slices:
        dy, dx = peak_slice
        x, y = dx.start, dy.start
        slice_data = data[peak_slice]
        index = np.argmax(slice_data)
        cx, cy = index//slice_data.shape[1], index % slice_data.shape[1]
        # cx, cy = centroidnp(data[peak_slice])
        centroids.append((x+cx, y+cy))
    return centroids
    

def Test(Code):

    Image = np.zeros((620, 580, 18), dtype='float32')
    thre = 0.001

    Points = []
    for c in range(Image.shape[-1]):
        for x in range(1):
            x = random.randint(0, 579)
            y = random.randint(0, 619)
            Image[y, x, c] = 1
            Points.append((c, x, y))

    for i in range(5):
        Image = gaussian_filter(Image, sigma=[5, 5, 0])

    Iters = 16

    if Code == 0:

        begin_time = time.time()

        for i in range(Iters):
            peaks = FindPeaks_2d(Image, 0.0001)
        print('the find cost %f seconds' % ((time.time()-begin_time)))
        
        # print(peaks)

    elif Code == 1:
        begin_time = time.time()

        for i in range(Iters):
            for j in range(Image.shape[-1]):
                temp = deepcopy(Image[:, :, j])
                peaks = FindPeaks_2d(temp, 0)

        print('the find cost %f seconds' % ((time.time()-begin_time)))

    elif Code == 2:
        begin_time = time.time() 

        for i in range(Iters):
            records = [[] for x in range(Image.shape[-1])]
            for j in range(Image.shape[-1]):
                
                temp = deepcopy(Image[:, :, j])
                # peaks = FindPeaks_2d(temp, 0)
                peaks = peaklables(temp, 0.001)
                records[j].append(peaks)

        print('the find cost %f seconds' % ((time.time()-begin_time)))

    elif Code == 3:
        Image = np.transpose(Image, [2, 0, 1])
        # Image = np.expand_dims(Image, axis=0)
        Image = np.tile(Image, (Iters, 1, 1, 1))
        Image = torch.from_numpy(Image)
        if torch.cuda.is_available():
            Image = Image.cuda()
        torch.cuda.empty_cache()
        begin_time = time.time()
        peaks = findpeaks_torch(Image, thre)
        # for i in range(Iters):
        #     # Image = np.transpose(Image, [2, 0, 1])
        #     peaks = findpeaks_torch(Image, thre)
        print('the find cost %f seconds' % ((time.time()-begin_time)))


if __name__ == "__main__":
    for i in range(4):
        Test(i)
    # Test(3)