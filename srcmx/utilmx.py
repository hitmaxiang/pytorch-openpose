'''
Description: the common used utilization functions
Version: 2.0
Autor: mario
Date: 2020-08-27 20:41:43
LastEditors: mario
LastEditTime: 2020-11-02 16:35:50
'''
import os
import re
import cv2
import matplotlib
import torch
from torch import nn
import torch.nn.functional as F
import scipy
import numpy as np
from tslearn import metrics
from scipy.io import loadmat
from numba import jit
from sklearn.preprocessing import scale


# @jit
def Calculate_shapelet_distance(query, timeseries):
    m_len = len(query)
    least_distance = np.linalg.norm(query - timeseries[:m_len])
    distance = least_distance
    best_loc = 0
    for loc in range(1, len(timeseries)-m_len+1):
        distance = np.linalg.norm(query - timeseries[loc:loc+m_len])
        if distance < least_distance:
            least_distance = distance
            best_loc = loc
    return least_distance, best_loc


def FaceDetect(frame):
    '''
    description: detect the face with the opencv trained model
    param: 
        frame: the BGR picture
    return: the facecenter and the face-marked frame
    author: mario
    '''
    face_cascade = cv2.CascadeClassifier()
    profileface_cascade = cv2.CascadeClassifier()

    # load the cascades
    face_cascade.load(cv2.samples.findFile('./model/haarcascade_frontalface_alt.xml'))
    profileface_cascade.load(cv2.samples.findFile('./model/haarcascade_profileface.xml'))

    # convert the frame into grayhist
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) == 0:
        faces = profileface_cascade.detectMultiScale(frame_gray)
    
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 0], 2)
    return faces, frame


def Write_frames_counts(dirname):
    '''
    description: get the infomation of the video framnumbers, write the file with the format
        ['videoname, framcounts, maximumframeinsubtitle']
    param: dirname, the filepath of the videofile
    return {type} 
    author: mario
    '''
    # extract the startframe information
    starfile = '../data/bbcposestartinfo.txt'
    startdict = {}
    # define the re pattern of the video index and startframe
    pattern1 = r'of\s*(\d+)\s*'
    pattern2 = r'is\s*(\d+).*$'
    with open(starfile) as f:
        lines = f.readlines()
        for line in lines:
            videoindex = int(re.findall(pattern1, line)[0])
            startindex = int(re.findall(pattern2, line)[0])
            startdict[videoindex] = startindex

    # extract the framecount info
    countdict = {}
    for i in range(1, 93):
        videopath = os.path.join(dirname, 'e%d.avi' % i)
        video = cv2.VideoCapture(videopath)
        if video.isOpened():
            count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            countdict[i] = count
        video.release()
    
    # get the maximaum frameindex from the dictinfo
    maxframdict = {}
    matfile = '../data/bbc_subtitles.mat'
    mat = loadmat(matfile)
    datas = mat['bbc_subtitles']

    for index, data in enumerate(datas[0]):
        if len(data[1][0]) == 0:
            maxframdict[index+1] = 0
        else:
            lastrecord = data[1][0][-1]
            maxframdict[index+1] = lastrecord[1][0][0]

    with open('../data/videoframeinfo.txt', 'w') as f:
        for i in list(countdict.keys()):
            f.write('e%d.avi\t%6d\t%6d\t%6d\n' % (i, startdict[i], countdict[i], maxframdict[i]))
            

class Records_Read_Write():
    def __init__(self):
        pass

    def Read_shaplets_cls_Records(self, filepath):
        '''
        description: extract the data (test data) from the record data
        param {type} 
        return {type} 
        author: mario
        '''
        RecordsDict = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
        params_pattern = r'word:\s*(\w+),.*lenth:\s*(\d+),\s*iters:\s*(\d+),\s*feature:\s*(\d+)'
        score_pattern = r'score:\s*(\d*.\d+)$'
        locs_pattern = r'Locs:(.*)$'
        for line in lines:
            params = re.findall(params_pattern, line)
            
            if len(params) != 0:
                word = params[0][0]
                args = params[0][1:]
                key = '-'.join(args)
                if word not in RecordsDict.keys():
                    RecordsDict[word] = {}
                    temp_dict = {}
                else:
                    if key not in RecordsDict[word].keys():
                        temp_dict = {}
                    else:
                        temp_dict = RecordsDict[word][key]

                if temp_dict == {}:
                    temp_dict = {'num': 1}
                    temp_dict['score'] = []
                    temp_dict['location'] = []
                else:
                    temp_dict['num'] += 1
                
                RecordsDict[word][key] = temp_dict
                continue

            score = re.findall(score_pattern, line)
            if len(score) != 0:
                RecordsDict[word][key]['score'].append(float(score[0]))
                continue
            
            locs = re.findall(locs_pattern, line)
            if len(locs) != 0:
                locs = locs[0].split()
                locs = [int(d) for d in locs]
                RecordsDict[word][key]['location'].append(locs)
            
        return RecordsDict
    
    def Write_shaplets_cls_Records(self, filepath, word, m_len, iters, featuremode, score, locs):

        with open(filepath, 'a+') as f:
            f.write('\nthe test of word: %s, with lenth: %d, iters: %d, feature: %d\n' %
                    (word, m_len, iters, featuremode))
            f.write('score:\t%f\n' % score)
            f.write('Locs:')
            for loc in locs:
                f.write('\t%d' % loc[0])
            f.write('\n\n')
    
    def Get_extract_ed_ing_files(self, datadir, init=False):
        filelists = []
        if init is False:
            with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                filelists.append(line)
        else:
            datafiles = os.listdir(datadir)
            with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'w') as f:
                for datafile in datafiles:
                    if os.path.splitext(datafile)[1] in ['.npy', 'pkl']:
                        f.write('%s\n' % datafile)
                        filelists.append(datafile)
        return filelists
    
    def Add_extract_ed_ing_files(self, datadir, addfilename):
        with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'a') as f:
            f.write('%s\n' % addfilename)


# handpose image drawed by opencv
def draw_handpose_by_opencv(canvas, peaks):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    for i in range(len(peaks)):
        for j in range(len(edges)):
            x1, y1 = peaks[i][edges[j][0]]
            x2, y2 = peaks[i][edges[j][1]]

            if (x1+y1 != 0) and (x2+y2 != 0):
                # cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([j/float(len(edges)), 1.0, 1.0])*255, thickness=2)
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color=[255, 0, 0], thickness=2)
        for k in range(len(peaks[i])):
            x, y = peaks[i][k]
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
    return canvas


def FindPeaks_2d(data, thre):
    assert len(data.shape) == 2
    map_left = np.zeros(data.shape)
    map_left[1:, :] = data[:-1, :]
    map_right = np.zeros(data.shape)
    map_right[:-1, :] = data[1:, :]
    map_up = np.zeros(data.shape)
    map_up[:, 1:] = data[:, :-1]
    map_down = np.zeros(data.shape)
    map_down[:, :-1] = data[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (data >= map_left, data >= map_right, data >= map_up, data >= map_down, data > thre))
    peaks_index = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse 
    return peaks_index


# @jit
def FindPeaks(data, thre):
    map_left = np.zeros(data.shape)
    map_left[1:, :] = data[:-1, :]
    map_right = np.zeros(data.shape)
    map_right[:-1, :] = data[1:, :]
    map_up = np.zeros(data.shape)
    map_up[:, 1:] = data[:, :-1]
    map_down = np.zeros(data.shape)
    map_down[:, :-1] = data[:, 1:]

    peaks_binary = np.logical_and.reduce(
        (data >= map_left, data >= map_right, data >= map_up, data >= map_down, data > thre))
    cordi = np.nonzero(peaks_binary)
    # peaks_index = list(zip(z[0], z[1], z[2]))  # note reverse 
    cordi = np.array(cordi).T
    if cordi.shape[-1] == 3:
        cordi = list(cordi)
        cordi.sort(key=lambda cor: cor[-1])
        counters = 0
        records = [[] for i in range(data.shape[-1])]
        for y, x, c in cordi:
            records[c].append((x, y, data[y, x, c], counters))
            counters += 1
        return records
    return cordi[:, ::-1]


def FindPeaks_2d_scipy(data, thre):
    data[data < thre] = 0
    labels, nums = scipy.ndimage.label(data)
    peak_slices = scipy.ndimage.find_objects(labels)
    centroids = []
    for peak_slice in peak_slices:
        dy, dx = peak_slice
        x, y = dx.start, dy.start
        slice_data = data[peak_slice]
        index = np.argmax(slice_data)
        cy, cx = (index+1)//slice_data.shape[1], index % slice_data.shape[1]
        # cx, cy = centroidnp(data[peak_slice])
        centroids.append((x+cx, y+cy))
    return centroids


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

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        # print("channels: ", channels.shape)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, 5, 5))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv1d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.weight, groups=self.channels)
        return x


if __name__ == "__main__":
    Records_Read_Write().Read_shaplets_cls_Records('../data/record.txt')