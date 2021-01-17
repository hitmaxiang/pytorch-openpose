import os
import sys
import cv2
import h5py
import json
import torch
import joblib
import numpy as np
import torch.utils.data as data_utl


def video_to_tensor(pic):
    '''convert a numpy.nadarry to tensor
    Args:
        pic (numpy.ndarry), the vieo clips with the shape of (T x H x W x C)
    Returns:
        the tensor of torch, with the shape of (C x T x H x W)
    '''

    return torch.from_numpy(np.transpose(pic, (3, 0, 1, 2)))


def Constructvideopathdict(videodir):
    videopathdict = {}
    filenames = os.listdir(videodir)

    for filename in filenames:
        name, ext = os.path.splitext(filename)
        if ext in ['.mp4', '.avi']:
            numkey = name[:3]
            videopathdict[numkey] = os.path.join(videodir, filename)
    
    return videopathdict


def load_rgb_frames_from_video(videonamedict, videonumkey, startnum, num, recpoint, resize=(256, 256)):
    
    videopath = videonamedict[videonumkey]
    videocap = cv2.VideoCapture(videopath)

    frames = []
    total_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))

    videocap.set(cv2.CAP_PROP_POS_FRAMES, startnum)
    for offset in range(min(num, int(total_frames-startnum))):
        ret, img = videocap.read()
        # crop 得到主要的 interpreter 的活动区域
        img = img[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]
        h, w, c = img.shape

        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        
        if w > 256 or h > 256:
            img = cv2.resize(img, dsize=(256, 256))
        
        img = (img / 255.) * 2 - 1

        frames.append(img)
    
    videocap.release()
    
    return np.array(frames, dtype=np.float32)


def make_dataset(samplepklfile, splitmode):
    sample_records = joblib.load(samplepklfile)
    num_classes = len(set(sample_records[:, 0]))
    dataset = []
    for record in sample_records:
        category, group, videonum, begidx, num = record
        if splitmode == 'train':
            if group not in [0, 1]:
                continue
        elif splitmode == 'test':
            if group != 2:
                continue
        # 构建 label 数据
        label = np.zeros((num, num_classes), np.float32)
        label[:, category] = 1

        dataset.append(videonum, begidx, num, label)
    
    return dataset


class BSLDataSet(data_utl.Dataset):

    def __init__(self, videodir, samplepklfile, splitmode, recpoint, transforms=None):
        self.data = make_dataset(samplepklfile, splitmode)
        self.videodir = videodir
        self.recpoint = recpoint
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        videonum, begidx, num, label = self.data[index]
        videonumkey = '03%d' % videonum
        videonumdict = Constructvideopathdict(self.videodir)

        imgs = load_rgb_frames_from_video(videonumdict, videonumkey, begidx, num, self.recpoint)

        fixlen = 32

        if len(imgs) < fixlen:
            imgs, label = self.pad(imgs, label)
        
        else:
            newbegin = np.random.randint(0, max(0, len(imgs)-fixlen)+1)
            imgs = imgs[newbegin:newbegin+fixlen]
            label = label[newbegin:newbegin+fixlen]
        
        if self.transforms is not None:
            imgs = self.transforms(imgs)
        
        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)

        return ret_img, ret_lab
    
    def pad(self, imgs, label, fixlen):
        if len(imgs) < fixlen:
            num_padding = fixlen - len(imgs)
            # 以 50% 的概率分别从前面或者后面进行补齐
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([pad, imgs], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs

        padded_label = np.tile(label[0], (fixlen, 1))

        return padded_imgs, padded_label
