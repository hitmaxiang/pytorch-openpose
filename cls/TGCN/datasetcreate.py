import sys
sys.path.append('../..')
import os
import h5py
import json
import torch
import joblib
import numpy as np
import torch.utils.data as data_utl

from srcmx import PreprocessingData
from sklearn import preprocessing


def load_pose_data(motionh5filepath, videonum, beginidx, endidx):
    with h5py.File(motionh5filepath, mode='r') as f:
        videokey = '03%d' % videonum
        posekey = 'posedata/pose/%s' % videokey
        handkey = 'handdata/hand/%s' % videokey

        posedata = f[posekey][beginidx:endidx]
        handdata = f[handkey][beginidx:endidx]

        pose = np.concatenate([posedata, handdata], axis=0)

        return pose


class PoseDataset(data_utl.Dataset):
    def __init__(self, motionh5filepath, sample_file, splimode, featuremode, num_frames):
        super().__init__()
        self.dataset = make_dataset(sample_file, splimode)
        self.motionh5filepath = motionh5filepath
        self.featuremode = featuremode
        self.num_frames = num_frames

        # set up the labelcoder
        words = sorted([x[0] for x in self.dataset])
        self.label_binarier = preprocessing.LabelBinarizer()
        self.label_binarier.fit(words)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        word, videonum, begidx, endidx = data
        posedata = load_pose_data(self.motionh5filepath, videonum, begidx, endidx)
        posedata = self.Random_select_frames(posedata, self.num_frames, 1)
        posedata = PreprocessingData.MotionJointFeatures(posedata, featuremode=self.featuremode)
        label = self.label_binarier.transform([word])

        return posedata, label

    def Random_select_frames(self, posedata, num_frames, randommode=0):
        N = len(posedata)
        if N > num_frames:
            if randommode == 0:
                # random start get a clip with length of num_frames
                candirange = num_frames - N
                begidx = np.random.randint(0, candirange+1)
                posedata = posedata[begidx:begidx+num_frames]
            elif randommode == 1:
                # select the clip with uniformly
                scale = N / num_frames
                indexes = np.arange(num_frames)
                indexes = round(scale * indexes)
                posedata = posedata[indexes]
        
        return posedata
    

def make_dataset(sample_file, splitmode):
    recordfile = h5py.File(sample_file, 'r')

    words = list(recordfile.keys())
    dataset = []
    for word in words:
        key = '%s/%s' % (word, splitmode)
        candi_samples = recordfile[key]
        for sample in candi_samples:
            videonum, begidx, endidx = sample
            dataset.append([word, videonum, begidx, endidx])
    return dataset