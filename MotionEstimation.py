import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from copy import deepcopy

from src import model
from src import util
from src.body import Body
from src.hand import Hand


# load the trained model of pose and hand 
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')


def ExtractMotionVideo(videopath, name, mode='body'):
    
    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print('the file %s is not exist' % videopath)
        return
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if mode == 'bodyhand':
        MotionMat = np.zeros((count, 60, 3))
        name = './data/%s-hb' % name
    else:
        MotionMat = np.zeros((count, 18, 3))
        name = './data/%s-bd' % name
    Recpoint = [(350, 100), (700, 400)]

    npyname = '%s.npy' % name
    if os.path.isfile(npyname):
        return

    index = 0
    while True:
        # randid = np.random.randint(count)
        # print('randid: %d' % randid)
        # video.set(cv2.CAP_PROP_POS_FRAMES, randid)
        ret, frame = video.read()
        # if ret is False or index > 40:
        if ret is False:
            break
        print('%d/%d' % (index, int(count)), end='\r')
        img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
        if mode == 'bodyhand':
            PoseMat = EstimationFrame(img, False)
        else:
            PoseMat = BodyFrame(img, False)
        MotionMat[index, :, :] = PoseMat
        index += 1
    np.save(name, MotionMat)
    print('')


# demonstrate mode
def EstimationVideo(inputfile):

    # open the video and read every frame data
    videofile = cv2.VideoCapture(inputfile)
    if not videofile.isOpened():
        raise Exception("the video file can not be open")
    
    # RECPOINT = ((400, 100), (640, 400))
    RECPOINT = [(350, 100), (700, 400)]
    COUNT = videofile.get(cv2.CAP_PROP_FRAME_COUNT)

    size = (RECPOINT[1][0]-RECPOINT[0][0], RECPOINT[1][1]-RECPOINT[0][1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoWriter = cv2.VideoWriter('body.avi', fourcc, 30, size)

    begin = time.time()
    number = 0
    while True:
        randframe = np.random.randint(COUNT)
        videofile.set(cv2.CAP_PROP_POS_FRAMES, randframe)
        # print(randframe)
        ret, frame = videofile.read()
        if ret is False:
            break
        print('%d/%d' % (number, int(COUNT)), end='\r')
        number += 1
        # cv2.rectangle(frame, RECPOINT[0], RECPOINT[1], [0, 0, 255], 2)
        img = frame[RECPOINT[0][1]:RECPOINT[1][1], RECPOINT[0][0]:RECPOINT[1][0], :]
        # cv2.imwrite('%d.jpg' % randframe, img)
        '''
        cv2.imshow('frame', img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        '''
        # begin = time.time()
        EstimationFrame(deepcopy(img, True))
        # VideoWriter.write(img)
        # print(time.time()-begin)
    videofile.release()
    VideoWriter.release()
    print('consum time %.2f' % (time.time()-begin))


def EstimationFrame(oriImg, display=False):
    candidate, subset = body_estimation(oriImg)
    PoseMat = np.zeros((60, 3))
    # find the most right person in the screen
    maxindex = None
    if len(subset) > 1:
        leftshoulderX = np.zeros((len(subset),))
        for person in range(len(subset)):
            index = int(subset[person][5])
            leftshoulderX[person] = candidate[index][0]
        maxindex = np.argmax(leftshoulderX)
        # clear the other person data
        for i in range(len(subset)):
            if i != maxindex:
                subset[i, :] = -1

    # get the choose person's keypoint
    if maxindex is not None:
        for keyindex in range(18):
            valueindex = int(subset[maxindex][keyindex])
            if valueindex != -1:
                PoseMat[keyindex, :] = candidate[valueindex][:3]

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        # if is_left:
        #     cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # else:
        #     cv2.rectangle(canvas, (x, y), (x+w, y+w), (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        '''
        if is_left:
            plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            plt.show()
        '''
        # print('the left hand', is_left)
        if not is_left:
            peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            PoseMat[39:39+21, :] = peaks
        else:
            peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], w-peaks[:, 0]-1+x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            PoseMat[18:18+21, :] = peaks
        #     print(peaks)
        all_hand_peaks.append(peaks[:, 0:2])
    
    # draw the body and hand 
    if display is True:
        canvas = deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        canvas = util.draw_handpose(canvas, all_hand_peaks)
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()

    return PoseMat


def BodyFrame(oriImg, display=False):
    candidate, subset = body_estimation(oriImg)
    BodyMat = np.zeros((18, 3))
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
                BodyMat[keyindex, :] = candidate[valueindex][:3]
    
    # draw the body and hand 
    if display is True:
        # get the data from the save file
        person = np.zeros((1, 18))
        position = np.zeros((18, 3))
        for i in range(18):
            if sum(BodyMat[i, :]) == 0:
                person[0, i] = -1
            else:
                person[0, i] = i
            position[i, :3] = BodyMat[i, :]

        canvas = deepcopy(oriImg)
        dataimg = deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        dataimg = util.draw_bodypose(dataimg, position, person)
        # plt.imshow(canvas[:, :, [2, 1, 0]])
        # plt.figure()
        # plt.imshow(dataimg[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()
        cv2.imshow('1', canvas)
        cv2.imshow('2', dataimg)
        cv2.waitKey(10)

    return BodyMat


def CheckNpydata(videopath, datapath):
    video = cv2.VideoCapture(videopath)
    PoseMat = np.load(datapath)
    if not video.isOpened():
        print('the file %s is not exist' % videopath)
        return
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if count != PoseMat.shape[0]:
        print('the count number is not equal')
    
    mode = 'bodyhand'
    if PoseMat.shape[1] == 18:
        mode = 'body'
    Recpoint = [(350, 100), (700, 400)]

    index = 0
    while True:
        ret, frame = video.read()
        if ret is False:
            break
        print('%d/%d' % (index, int(count)), end='\r')
        img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
        pose = PoseMat[index, :, :]

        if mode == 'body':
            subset = np.zeros((1, 20))
            candidate = np.zeros((20, 4))
            for i in range(18):
                if sum(pose[i, :]) == 0:
                    subset[0, i] = -1
                else:
                    subset[0, i] = i
                candidate[i, :3] = pose[i, :]
            canvas = util.draw_bodypose(img, candidate, subset)
            cv2.imshow('img', canvas)
            key = cv2.waitKey(30)
            if key & 0xff == ord('q'):
                break
        else:
            pass
        index += 1


if __name__ == "__main__":
    import os
    import sys
    args = sys.argv[1:]
    begin = 0
    for arg in args:
        begin = int(arg)

    destfolder = '/home/mario/sda/signdata/bbcpose'
    # destfolder = '/home/hit605/public/Upload/mx/bbcpose'
    filenames = os.listdir(destfolder)

    Code = 2
    if Code == 1:  # random choose one file to estimation
        filename = filenames[np.random.randint(len(filenames))]
        filepath = os.path.join(destfolder, filename)
        EstimationVideo(filepath)
    elif Code == 2:
        filenames.sort()
        end = min(len(filenames), begin+10)
        for filename in filenames[begin:end]:
            filepath = os.path.join(destfolder, filename)
            name = filename.split('.')[0]
            print(name)
            ExtractMotionVideo(filepath, name, mode='body')
    elif Code == 3:
        filenames.sort()
        num = np.random.randint(len(filenames))
        for filename in filenames[num:]:
            filepath = os.path.join(destfolder, filename) 
            name = filename.split('.')[0]
            datapath = './data/%s-bd.npy' % name
            if os.path.isfile(datapath):
                print(name)
                CheckNpydata(filepath, datapath)