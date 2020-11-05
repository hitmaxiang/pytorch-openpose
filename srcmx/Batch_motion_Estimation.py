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


body_pth = '../model/body_pose_model.pth'
hand_pth = '../model/hand_pose_model.pth'

# Batch_Body_model = BM.Batch_body(body_pth)


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
    
    batch_size = 16
    # get the video-names with the giving videofolder
    filenames = os.listdir(videofolder)
    filenames.sort()

    if code == 0:
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
    Batch_Body_model = BM.Batch_body(body_pth)

    print(args.testcode, args.init, args.mode, args.dataset, args.server)
    Test(code=args.testcode, init=args.init, mode=args.mode,
         dataset=args.dataset, server=args.server)
