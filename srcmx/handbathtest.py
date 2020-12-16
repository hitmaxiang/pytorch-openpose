import sys
sys.path.append('..')

import os 
import cv2
import time
import torch
import joblib
import utilmx
import h5py
# import torchvision
import torch.nn.functional as F
import numpy as np
import argparse
from src import util
from copy import deepcopy
from src.model import handpose_model
from skimage.measure import label
from scipy.ndimage.filters import gaussian_filter

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Batch_model import HandImageDataset, Batch_hand


class HandImageDataset_mx(Dataset):
    def __init__(self, videopath, motiondata, recpoints, boxsize):
        super().__init__()
        self.video = cv2.VideoCapture(videopath)
        self.motiondata = motiondata
        self.crop = recpoints
        self.boxsize = boxsize

        self.FrameCounts = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        assert self.FrameCounts == len(self.motiondata)
    
    def __len__(self):
        return self.FrameCounts
    
    def __getitem__(self, idx):
        _, image = self.video.read()
        image = image[self.crop[0][1]:self.crop[1][1], self.crop[0][0]:self.crop[1][0], :]

        # 根据posemat 构建 subset 
        pose = self.motiondata[idx, :, :]
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(pose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = pose[i, :2]
        
        hands_list = utilmx.handDetect(candidate, subset, image.shape[:2])

        LeftHand = np.zeros((self.boxsize, self.boxsize, 3), dtype=np.uint8) + 128
        RightHand = np.zeros_like(LeftHand) + 128
        leftparams = np.zeros((3,))
        rightparams = np.zeros_like(leftparams)

        for x, y, w, is_left in hands_list:
            if not is_left:
                tempimg = image[y:y+w, x:x+w, :]
                RightHand = cv2.resize(tempimg, (self.boxsize, self.boxsize), interpolation=cv2.INTER_CUBIC)
                rightparams = np.array([x, y, w])
            else:
                tempimg = cv2.flip(image[y:y+w, x:x+w, :], 1)
                LeftHand = cv2.resize(tempimg, (self.boxsize, self.boxsize), interpolation=cv2.INTER_CUBIC)
                leftparams = np.array([x, y, w])

        RightHand = transforms.ToTensor()(RightHand)
        LeftHand = transforms.ToTensor()(LeftHand)

        return LeftHand, leftparams, RightHand, rightparams


class Batch_hand_mx():
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = util.transfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # self.scale_search = [0.5, 1.0, 1.5, 2.0]
        self.scale_search = [1.0]
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128
        self.thre = 0.035

        self.guassian_filter_conv = utilmx.GaussianBlurConv(22)
        if torch.cuda.is_available():
            self.guassian_filter_conv.cuda()

    def __call__(self, batch_imgs):
        
        # t0 = time.time()
        
        # h0, w0 = oriImg.shape[:2]
        # assert h0 == w0
        # s0 = h0/self.boxsize
        # oriImg = cv2.resize(oriImg, (self.boxsize, self.boxsize), interpolation=cv2.INTER_CUBIC)

        # batch_imgs = np.transpose(np.float32(oriImg[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256
        # batch_imgs = torch.from_numpy(batch_imgs)
    
        # if np.random.randint(10) > 5:
        #     print('zero nimage')
        #     batch_imgs = torch.zeros_like(batch_imgs)

        batch_size, c, h, w = batch_imgs.size()
        # heatmap_avg = torch.zeros(batch_size, 22, h, w)

        if torch.cuda.is_available():
            batch_imgs = batch_imgs.cuda()
            # heatmap_avg = heatmap_avg.cuda()
        with torch.no_grad():
            # for s in self.scale_search:
                # scale, n_h, n_w, pad_h, pad_w = self.calculate_size_pad(s, h, w)
                # T_images = F.interpolate(batch_imgs, scale_factor=scale, mode='bicubic')
                # T_images = F.pad(T_images, [0, pad_w, 0, pad_h], mode='constant', value=0)
            batch_imgs = batch_imgs - 0.5
            heatmap = self.model(batch_imgs)
            heatmap = F.interpolate(heatmap, scale_factor=self.stride, mode='bicubic')
            # heatmap = heatmap[:, :, :n_h, :n_w]
            # heatmap = F.interpolate(heatmap, size=(h, w), mode='bicubic')

            heatmap = self.guassian_filter_conv(heatmap)
            # Binary = heatmap > self.thre
            
            heatmap = heatmap.cpu().numpy()
            # Binary = Binary.cpu().numpy()

            heatmap = np.transpose(heatmap, (0, 2, 3, 1))
            # Binary = np.transpose(Binary, (0, 2, 3, 1))
        
        Batch_peaks = []
        for i in range(batch_size):
            # heatmap_avg = heatmap[i]
            all_peaks = []
            for part in range(21):
                map_ori = heatmap[i, :, :, part]
                binary = map_ori > self.thre
                # one_heatmap = gaussian_filter(map_ori, sigma=3)
                # binary = np.ascontiguousarray(one_heatmap > self.thre, dtype=np.uint8)
                # 全部小于阈值
                if np.sum(binary) == 0:
                    all_peaks.append([0, 0, 0])
                    continue
                label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
                max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
                # label_img[label_img != max_index] = 0
                # map_ori[label_img == 0] = 0
                map_ori[label_img != max_index] = 0

                y, x = util.npmax(map_ori)
                all_peaks.append([x, y, np.max(map_ori)])
                # all_peaks.append([int(round(x*s0)), int(round(y*s0)), np.max(map_ori)])
            Batch_peaks.append(all_peaks)


        # # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
        # 
        # for m in range(len(multiplier)):
        #     scale = multiplier[m]
        #     imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        #     imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
        #     im = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        #     im = np.ascontiguousarray(im)

        #     data = torch.from_numpy(im).float()
        #     if torch.cuda.is_available():
        #         data = data.cuda()
        #     # data = data.permute([2, 0, 1]).unsqueeze(0).float()
        #     with torch.no_grad():
        #         output = self.model(data).cpu().numpy()
        #         # output = self.model(data).numpy()q

        #     # extract outputs, resize, and remove padding
        #     heatmap = np.transpose(np.squeeze(output), (1, 2, 0))  # output 1 is heatmaps
        #     heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        #     heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        #     heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        #     heatmap_avg += heatmap / len(multiplier)

        # print('1:', time.time()-t0)
        # t0 = time.time()

        # all_peaks = []
        # for part in range(21):
        #     map_ori = heatmap_avg[:, :, part]
        #     one_heatmap = gaussian_filter(map_ori, sigma=3)
        #     binary = np.ascontiguousarray(one_heatmap > self.thre, dtype=np.uint8)
        #     # 全部小于阈值
        #     if np.sum(binary) == 0:
        #         all_peaks.append([0, 0, 0])
        #         continue
        #     label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
        #     max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
        #     # label_img[label_img != max_index] = 0
        #     # map_ori[label_img == 0] = 0
        #     map_ori[label_img != max_index] = 0

        #     y, x = util.npmax(map_ori)
        #     # all_peaks.append([x, y, np.max(map_ori)])
        #     all_peaks.append([int(round(x*s0)), int(round(y*s0)), np.max(map_ori)])
            
        # # print('find:', time.time()-t0)
        # t0 = time.time()
        return np.array(Batch_peaks)

    def calculate_size_pad(self, g_scale, height, width):
        scale = self.boxsize * g_scale/height
        h, w = int(height*scale), int(width*scale)
        pad_h = (self.stride - (h % self.stride)) % self.stride
        pad_w = (self.stride - (w % self.stride)) % self.stride
        return (scale, h, w, pad_h, pad_w)
# load the trained model of pose and hand 
# body_estimation = Body('../model/body_pose_model.pth')


def detecthand(videopath, PoseMat, recpoint, display=False):
    video = cv2.VideoCapture(videopath)
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if count != PoseMat.shape[0]:
        print('the count number is not equal')

    index = 0
    counters = 0
    validkeypointnum = 0
    Maxiter = 400
    while True:
        
        if counters > Maxiter:
            break
        index = np.random.randint(count)
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        counters += 1
        cv2.waitKey(300)

        ret, frame = video.read()
        if ret is False:
            break
        print('%d/%d' % (index, int(count)), end='\r')
        t = time.time()
        oriImg = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]

        # 根据posemat 构建 subset 
        pose = PoseMat[index, :, :]
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(pose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = pose[i, :2]
        
        hands_list = utilmx.handDetect(candidate, subset, oriImg.shape[:2])
        # print(time.time()-t)
        all_hand_peaks = []
        canvas = deepcopy(oriImg)
        for x, y, w, is_left in hands_list:
            # if is_left:
            #     cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # else:
            #     cv2.rectangle(canvas, (x, y), (x+w, y+w), (255, 0, 0), 2, lineType=cv2.LINE_AA)
            if not is_left:
                peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            else:
                peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], w-peaks[:, 0]-1+x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            all_hand_peaks.append(peaks[:, 0:2])
        print(time.time()-t)

        # draw the body and hand 
        if display:
            canvas = util.draw_bodypose(canvas, candidate, subset)
            canvas = utilmx.draw_handpose_by_opencv(canvas, all_hand_peaks)
            # # cv2.imwrite('%d.jpg' % counters, canvas)
            cv2.imshow('bodyhand', canvas)
            q = cv2.waitKey(0) & 0xff
            if q == ord('q'):
                break
        # 计算有效点的个数
        for peaks in all_hand_peaks:
            for point in peaks:
                if point[0] + point[1] != 0:
                    validkeypointnum += 1
        
    print('the avg peaks is %f' % (validkeypointnum/counters))


def BatchHandExtract(videopath, motiondata, recpoints, outpath):
    boxsize = 368
    batchsize = 32

    Handdataset = HandImageDataset(videopath, motiondata, recpoints, boxsize)
    HandDataloader = DataLoader(Handdataset, batchsize, shuffle=False)

    HandMat = np.zeros((len(motiondata), 42, 3))
    count = 0
    for samples in HandDataloader:
        t0 = time.time()

        if count > 200:
            joblib.dump(HandMat[:count], outpath)
            break

        LeftHand, leftparams, RightHand, rightparams = samples
        Leftpeaks = hand_estimation(LeftHand)
        rightpeaks = hand_estimation(RightHand)
        
        leftparams = leftparams.numpy()
        rightparams = rightparams.numpy()

        for i in range(batchsize):
            # right peaks
            rpeaks = rightpeaks[i]
            rx, ry, rw = rightparams[i]
            rpeaks[:, :2] = rpeaks[:, :2] * rw / boxsize

            rpeaks[:, 0] = np.where(rpeaks[:, 0] == 0, rpeaks[:, 0], rpeaks[:, 0]+rx)
            rpeaks[:, 1] = np.where(rpeaks[:, 1] == 0, rpeaks[:, 1], rpeaks[:, 1]+ry)
            HandMat[count, 21:, :] = rpeaks

            # left peaks
            lpeaks = Leftpeaks[i]
            lx, ly, lw = leftparams[i]
            lpeaks[:, :2] = lpeaks[:, :2] * lw / boxsize
            lpeaks[:, 0] = np.where(lpeaks[:, 0] == 0, lpeaks[:, 0], lw-lpeaks[:, 0]-1+lx)
            lpeaks[:, 1] = np.where(lpeaks[:, 1] == 0, lpeaks[:, 1], lpeaks[:, 1]+ly)
            HandMat[count, :21, :] = lpeaks

            count += 1
            # if count % 1000 == 0:
            #     print('%s-%d/%d' % (outpath, count, len(motiondata)))
        diff = time.time() - t0
        print('count: %d cost %f--avg: %f' % (count, diff, diff/batchsize))


def CheckHandData(videopath, motiondata, handdata, recpoint):
    video = cv2.VideoCapture(videopath)
    for i in range(len(handdata)):
        _, frame = video.read()

        hands = handdata[i]
        lpeaks = hands[:21, :2]
        rpeaks = hands[21:, :2]

        oriImg = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]

        # 根据posemat 构建 subset 
        pose = motiondata[i, :, :]
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(pose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = pose[i, :2]
        
        canvas = util.draw_bodypose(oriImg, candidate, subset)
        canvas = utilmx.draw_handpose_by_opencv(canvas, [lpeaks, rpeaks])
        # cv2.imwrite('%d.jpg' % counters, canvas)
        cv2.imshow('bodyhand', canvas)
        q = cv2.waitKey(30) & 0xff
        if q == ord('q'):
            break
        

def Test(code, display):
    global hand_estimation
    # hand_estimation = Batch_hand_mx('../model/hand_pose_model.pth')
    hand_estimation = Batch_hand('../model/hand_pose_model.pth')

    # videofolder = '/home/mario/signdata/spbsl/normal'
    videofolder = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'

    motiondatadir = '../data/spbsl/motionsdic.pkl'
    motionhdf5filepath = '../data/spbsl/motiondata.hdf5'

    Recpoint = [(700, 100), (1280, 720)]
    
    filenames = os.listdir(videofolder)
    filenames.sort()

    if TestCode == 0:
        motiondatadic = joblib.load(motiondatadir)
        np.random.shuffle(filenames)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                keynum = filename[:3]
                PoseMat = motiondatadic[keynum][0]
                detecthand(filepath, PoseMat, Recpoint)
    elif TestCode == 1:
        np.random.shuffle(filenames)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                keynum = filename[:3]
                if keynum == '100':
                    if os.path.exists('../data/spbsl/100.pkl'):
                        PoseMat = joblib.load('../data/spbsl/100.pkl')
                    else:
                        motiondatadic = joblib.load(motiondatadir)
                        PoseMat = motiondatadic[keynum][0]
                        joblib.dump(PoseMat, '../data/spbsl/100.pkl')
                    detecthand(filepath, PoseMat, Recpoint, display=display)
    elif TestCode == 2:
        wantnum = '013'
        wantmotionpath = '../data/spbsl/%s.pkl' % wantnum
        wanthandpath = '../data/spbsl/video-%s-hand.pkl' % wantnum

        np.random.shuffle(filenames)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                keynum = filename[:3]
                if keynum == wantnum:
                    if os.path.exists(wantmotionpath):
                        PoseMat = joblib.load(wantmotionpath)
                    else:
                        motiondatadic = joblib.load(motiondatadir)
                        PoseMat = motiondatadic[keynum][0]
                        joblib.dump(PoseMat, wantmotionpath)
                    if not os.path.exists(wanthandpath):
                        BatchHandExtract(filepath, PoseMat, Recpoint, wanthandpath)
                    
                    handMat = joblib.load(wanthandpath)

                    CheckHandData(filepath, PoseMat, handMat, Recpoint)
    
    elif TestCode == 3:
        # check the extracted and saved data of hdf5 format
        # wantnum = '013'
        # wantmotionpath = '../data/spbsl/%s.pkl' % wantnum
        # wanthandpath = '../data/spbsl/video-%s-hand.pkl' % wantnum

        np.random.shuffle(filenames)
        motionhdf5 = h5py.File(motionhdf5filepath, 'r')

        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                keynum = filename[:3]
                posekey = 'posedata/pose/%s' % keynum
                handkey = 'handdata/hand/%s' % keynum
                print(keynum)
                pose = motionhdf5[posekey][:]
                hand = motionhdf5[handkey][:]
                CheckHandData(filepath, pose, hand, Recpoint)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testcode', type=int, help='the test code', default=3)
    parser.add_argument('-d', '--display', action='store_true')
    args = parser.parse_args()

    TestCode = args.testcode
    Display = args.display

    Test(TestCode, Display)