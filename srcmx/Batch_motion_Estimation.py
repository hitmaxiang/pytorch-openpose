import sys
sys.path.append('..')
import cv2
import torch
import os
import argparse
import time
import joblib
import utilmx
import numpy as np
import Batch_model as BM
from torch.utils.data import DataLoader

body_pth = '../model/body_pose_model.pth'
hand_pth = '../model/hand_pose_model.pth'


# Batch_Body_model = BM.Batch_body(body_pth)
def Batch_hand_extraction(videopath, motiondata, recpoints, outpath):
    boxsize = 368
    batchsize = 32

    Handdataset = BM.HandImageDataset(videopath, motiondata, recpoints, boxsize)
    HandDataloader = DataLoader(Handdataset, batchsize, shuffle=False)

    COUNTS = len(motiondata)
    HandMat = np.zeros((COUNTS, 42, 3))
    count = 0
    for samples in HandDataloader:

        LeftHand, leftparams, RightHand, rightparams = samples
        Leftpeaks = batch_hand_estimation(LeftHand)
        rightpeaks = batch_hand_estimation(RightHand)
        
        leftparams = leftparams.numpy()
        rightparams = rightparams.numpy()

        # if count > 1000:
        #     joblib.dump(HandMat[:count], outpath)
        #     return

        for i in range(len(Leftpeaks)):
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
            if count % 1000 == 0:
                print('%s-%d/%d' % (outpath, count, COUNTS))

    joblib.dump(HandMat, outpath)
    print('the %s file is saved' % outpath)
            

def Batch_body_extraction(videopath, outpath, batch_size, recpoint):

    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print('the file %s is not exist' % videopath)
        return
    COUNTS = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    video_dataloader = BM.GetVideoDataLoader(videopath, batch_size, recpoint)

    outname = os.path.split(outpath)[1]
    # COUNTS = len(video_dataloader)
    MotionMat = np.zeros((COUNTS, 18, 3))
    count = 0
    for batch_images in video_dataloader:
        # real_batch_size = batch_images.size[0]
        results = Batch_Body_model(batch_images)

        for i in range(len(results)):
            PoseMat = np.zeros((18, 3))
            candidate, subset = results[i]
            # find the most right person in the screen

            maxindex = None
            if len(subset) >= 1:
                leftshoulderX = np.zeros((len(subset),))
                for person in range(len(subset)):
                    index = int(subset[person][5])
                    leftshoulderX[person] = candidate[index][0]
                maxindex = np.argmax(leftshoulderX)
            
            # get the choose person's keypoint
            if maxindex is not None:
                for keyindex in range(18):
                    valueindex = int(subset[maxindex][keyindex])
                    if valueindex != -1:
                        PoseMat[keyindex, :] = candidate[valueindex][:3]
            
            MotionMat[count] = PoseMat
            count += 1
        
            if count % 1000 == 0:
                print('%s-%d/%d' % (outname, count, COUNTS))
    joblib.dump(MotionMat, outpath)
    print('the %s file is saved' % outpath)


def Test(code, init=False, mode='body', dataset='spbsl', server=False):
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
    
    # get the video-names with the giving videofolder
    filenames = os.listdir(videofolder)
    filenames.sort()

    if code == 0:
        global Batch_Body_model
        Batch_Body_model = BM.Batch_body(body_pth)
        batch_size = 32
        print('batch extract the motion data from the videos and save')
        np.random.shuffle(filenames)
        utilmx.Records_Read_Write().Get_extract_ed_ing_files(datadir, init)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                complet_files = utilmx.Records_Read_Write().Get_extract_ed_ing_files(datadir)
                complet_videos = [x[:9] for x in complet_files]
                filepath = os.path.join(videofolder, filename)
                outname = 'video-%s-%s.pkl' % (filename[:3], mode)
                if mode == 'body' and (outname[:9] not in complet_videos):
                    print(outname)
                    utilmx.Records_Read_Write().Add_extract_ed_ing_files(datadir, outname)
                    outpath = os.path.join(datadir, outname)
                    Batch_body_extraction(filepath, outpath, batch_size, Recpoint)
                elif outname not in complet_files:
                    pass
    elif code == 1:
        global batch_hand_estimation
        batch_hand_estimation = BM.Batch_hand(hand_pth)
        batch_size = 32
        motiondatadicpath = '../data/spbsl/motionsdic.pkl'
        motiondict = joblib.load(motiondatadicpath)

        print('batch extract the hand data from the videos and save')
        np.random.shuffle(filenames)
        utilmx.Records_Read_Write().Get_extract_ed_ing_files(datadir, init)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                complet_files = utilmx.Records_Read_Write().Get_extract_ed_ing_files(datadir)
                complet_videos = [x[:9] for x in complet_files]
                filepath = os.path.join(videofolder, filename)
                outname = 'video-%s-hand.pkl' % (filename[:3])
                if outname[:9] not in complet_videos:
                    keynum = filename[:3]
                    motiondata = motiondict[keynum][0]
                    print(outname)
                    utilmx.Records_Read_Write().Add_extract_ed_ing_files(datadir, outname)
                    outpath = os.path.join(datadir, outname)
                    Batch_hand_extraction(filepath, motiondata, Recpoint, outpath)
                elif outname not in complet_files:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('testcode', type=int, help='the test code')
    parser.add_argument('-i', '--init', action='store_true')
    parser.add_argument('-m', '--mode', choices=['body', 'handbody'], default='body')
    parser.add_argument('-d', '--dataset', choices=['spbsl', 'bbc'], default='spbsl')
    parser.add_argument('-s', '--server', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, choices=[0, 1], default=0)

    args = parser.parse_args()
    cuda = args.cuda
    if cuda == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print('cuda 1')
    else:
        print('cuda 0')

    print(args.testcode, args.init, args.mode, args.dataset, args.server)
    Test(code=args.testcode, init=args.init, mode=args.mode,
         dataset=args.dataset, server=args.server)
