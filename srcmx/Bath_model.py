'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-11-02 09:00:02
LastEditors: mario
LastEditTime: 2020-11-02 21:54:48
'''
import sys
sys.path.append('..')

import os
import cv2
import math
import time
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TTF

from numba import jit
import utilmx

from copy import deepcopy
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.filters import gaussian_filter

from src import util
from skimage.measure import label
from src.model import handpose_model
from src.model import bodypose_model


class VideoDataset(Dataset):
    def __init__(self, videopath, crop=None, transform=None):
        super().__init__()
        self.video = cv2.VideoCapture(videopath)
        self.crop = crop
        self.transform = transform
    
    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __getitem__(self, idx):
        _, image = self.video.read()
        if self.crop is not None:
            image = image[self.crop[0][1]:self.crop[1][1], self.crop[0][0]:self.crop[1][0], :]
        if self.transform is not None:
            image = self.transform(image)
        return image


class Batch_body():
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # define some variables
        self.scale_search = 0.5
        self.boxsize = 368
        self.stride = 8
        self.padvalue = 128
        self.thre1 = 0.1
        self.thre2 = 0.05

        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                        [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                       [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                       [55, 56], [37, 38], [45, 46]]
    
    def __call__(self, batch_images):

        batch_size, c, h, w = batch_images.size()
        scale, n_h, n_w, pad_h, pad_w = self.calculate_size_pad(self.scale_search, h, w)

        begin_time = time.time()

        if torch.cuda.is_available():
            batch_images = batch_images.cuda()
        with torch.no_grad():
            # prepare the input data of the model
            batch_images = F.interpolate(batch_images, scale_factor=scale, mode='bicubic')
            batch_images = batch_images - 0.5
            batch_images = F.pad(batch_images, [0, pad_w, 0, pad_h], mode='constant', value=0)
            
            # estimate with the model
            Mconv7_stage6_L1, Mconv7_stage6_L2 = self.model(batch_images)

            # process the outdata of the model for the following use
            b_heatmap = F.interpolate(Mconv7_stage6_L2, scale_factor=self.stride, mode='bicubic')
            b_heatmap = b_heatmap[:, :, :n_h, :n_w]
            b_heatmap = F.interpolate(b_heatmap, size=(h, w), mode='bicubic')

            b_paf = F.interpolate(Mconv7_stage6_L1, scale_factor=self.stride, mode='bicubic')
            b_paf = b_paf[:, :, :n_h, :n_w]
            b_paf = F.interpolate(b_paf, size=(h, w), mode='bicubic')

            # move the data from the cuda to cpu 
            b_heatmap = b_heatmap.cpu().numpy()
            b_paf = b_paf.cpu().numpy()
            torch.cuda.empty_cache()
        
        # rearrange the axises of the data
        b_heatmap = np.transpose(b_heatmap, [0, 2, 3, 1])
        b_paf = np.transpose(b_paf, [0, 2, 3, 1])

        print('the data --> model --> cost %f seconds' % (time.time()-begin_time))

        begin_time = time.time()
        results = []
        for batch_id in range(len(b_heatmap)):
            candidates, subset = self.FindBody_frame(b_heatmap[batch_id], b_paf[batch_id])
            results.append((candidates, subset))
        print('the findprocess cost %f seconds' % (time.time()-begin_time))
        return results
    
    def FindBody_frame(self, heatmap, paf):
        all_peaks = []
        peak_counter = 0
        # begin_time = time.time()
        # heatmap_gf = gaussian_filter(heatmap, sigma=[3, 3, 0])
        heatmap_gf = cv2.GaussianBlur(heatmap, ksize=(0, 0), sigmaX=3, sigmaY=3)
        for part in range(18):
            map_ori = heatmap[:, :, part]
            one_heatmap = deepcopy(heatmap_gf[:, :, part])
            peaks = utilmx.FindPeaks_2d_scipy(one_heatmap, self.thre1)
            # peaks = utilmx.FindPeaks(one_heatmap, self.thre1)
            # peaks = utilmx.FindPeaks_2d(one_heatmap, self.thre1)

            peaks_with_score = [(x[0], x[1], map_ori[x[1], x[0]],) for x in peaks]
            peak_id = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (peak_id[i],) for i in range(len(peak_id))]
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        
        connection_all = []
        special_k = []
        mid_num = 10
        mapIdx = self.mapIdx
        limbSeq = self.limbSeq
        for k in range(len(mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # add a small value to avoid divide error
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-10  
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[k][1])), int(round(startend[k][0])), 0]
                                          for k in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[k][1])), int(round(startend[k][0])), 1]
                                          for k in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * heatmap.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        # print('the find-2 cost %f seconds' % (time.time()-begin_time))

        # begin_time = time.time()
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        # print('the find-3 cost %f seconds' % (time.time()-begin_time))

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset

    def calculate_size_pad(self, g_scale, height, width):
        scale = self.boxsize * g_scale/height
        h, w = int(height*scale), int(width*scale)
        pad_h = (self.stride - (h % self.stride)) % self.stride
        pad_w = (self.stride - (w % self.stride)) % self.stride
        return (scale, h, w, pad_h, pad_w)
        

def GetVideoDataLoader(videopath, bath_size, recpoint):
    video_dataset = VideoDataset(videopath, crop=recpoint, transform=transforms.ToTensor())
    video_dataloder = DataLoader(video_dataset, bath_size, shuffle=False)
    return video_dataloder


def Show_images(batch_images, windowname='video'):
    batch_images = batch_images.numpy()
    batch_images = np.transpose(batch_images, [0, 2, 3, 1])
    for i in range(len(batch_images)):
        cv2.imshow(windowname, batch_images[i])
        key = cv2.waitKey(5) & 0xff
        if key == ord('q'):
            return False
    return True


def Show_Bodys(batch_images, results):
    batch_images = batch_images * 255
    # batch_images = batch_images.astype(np.uint8)
    batch_images = torch.clamp(batch_images, 0, 255)
    batch_images = batch_images.numpy().astype(np.uint8)
    batch_images = np.transpose(batch_images, [0, 2, 3, 1])
    for i in range(len(batch_images)):
        canvas = deepcopy(batch_images[i])
        canvas = np.ascontiguousarray(canvas)
        candidate, subset = results[i]
        canvas = util.draw_bodypose(canvas, candidate, subset)
        cv2.imshow('bodypose', canvas)
        key = cv2.waitKey(5) & 0xff
        if key == ord('q'):
            return False
    return True


def Test(testcode, dataset='spbsl', server=True):
    body_pth = '../model/body_pose_model.pth'
    hand_pth = '../model/hand_pose_model.pth'
    
    if dataset == 'spbsl':
        if server is False:
            videofolder = '/home/mario/sda/signdata/Scottish parliament/bsl-cls/normal'
        else:
            videofolder = '/home/mario/signdata/spbsl/normal'

        datadir = '../data/spbsl'
        Recpoint = [(700, 100), (1280, 720)]

    elif dataset == 'bbc':
        if server is False:
            videofolder = '/home/mario/sda/signdata/bbcpose'
        else:
            videofolder = '/home/mario/signdata/bbc'
        datadir = '../data/bbc'
        Recpoint = [(350, 100), (700, 400)]
    
    Batch_Size = 16
    filenames = os.listdir(videofolder)
    filenames.sort()

    if testcode == 0:
        print('verify the data read is consective')
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videofolder, filename)
                video_dataloader = GetVideoDataLoader(videopath, Batch_Size, Recpoint)
                for i, batch_imgs in enumerate(video_dataloader):
                    print(batch_imgs.size())
                    if not Show_images(batch_imgs):
                        break
                break
    elif testcode == 1:
        print('test the batch body')
        bath_body_model = Batch_body(body_pth)
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videofolder, filename)
                video_dataloader = GetVideoDataLoader(videopath, Batch_Size, Recpoint)
                for i, batch_images in enumerate(video_dataloader):
                    # batch_images = batch_images.numpy()
                    # batch_images = np.transpose(batch_images, [0, 2, 3, 1])
                    begin_time = time.time()
                    results = bath_body_model(deepcopy(batch_images))
                    print('each image extraction cost %f seconds\n\n' % ((time.time()-begin_time)/Batch_Size))
                    if not Show_Bodys(batch_images, results):
                        break



if __name__ == "__main__":
    Test(1)
    

