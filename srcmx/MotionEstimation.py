import sys
sys.path.append('..')
import cv2
import os
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src import model
from src import util
from src.body import Body
from src.hand import Hand

# load the trained model of pose and hand 
body_estimation = Body('../model/body_pose_model.pth')
hand_estimation = Hand('../model/hand_pose_model.pth')


def Extract_MotionData_from_Video(videopath, outname, mode='body', overwrite=False):
    '''
    description: Extract the motion data from video and save as a data file
    param:
        videopath:(str), the input path of videofile
        outname:(str) the output name of the saved data
        mode:(str) the mode of the motiondata, 'body' means only skeleton data, and
            'bodyhand' means that the skeleton and hand data will be extracted
        overwrite:(bool), indicate whether to overwrite the exist save data
    return {dataarray}: the motion data array with the shape of (framecount, njoints, ndims),
        and framecount is the number of the video, njoints is the joints of the human, 'body' with
        18 joints, and 'bodyhand' with  18 + 21*2=60, ndims means (x_pos, y_pos, c_score), represent 
        the positions of the joint, and the c_sore represent the condidence score
    author: mario
    '''
    
    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print('the file %s is not exist' % videopath)
        return
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if mode == 'bodyhand':
        MotionMat = np.zeros((count, 60, 3))
        outname = '../data/%s-hb' % outname
    else:
        MotionMat = np.zeros((count, 18, 3))
        outname = '../data/%s-bd' % outname
    # the user-defined ROI
    Recpoint = [(350, 100), (700, 400)]

    npyname = '%s.npy' % outname
    if os.path.isfile(npyname) and overwrite is False:
        return

    index = 0
    while True:
        ret, frame = video.read()
        # if ret is False or index > 40:
        if ret is False:
            break
        # print the progress information every 100 frame
        if index % 100 == 0:
            print('%s-%d/%d' % (name, index, int(count)))
        # crop the ROI of the frame
        img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
        PoseMat = MotionData_every_frame(img, mode, display=False)
        MotionMat[index, :, :] = PoseMat
        index += 1
    np.save(npyname, MotionMat)
    print('%s is saved!')


def Demo_motion_of_Video(inputfile, mode):
    '''
    description: demonstrate the extract data of the inputfile
    param:
        inputfile(str): the path of the input video
        mode(str): the 'body' mode(skeleton) or 'bodyhand' mode(skeleton + hands)
    return: None
    author: mario
    '''
    # open the video and read every frame data
    videofile = cv2.VideoCapture(inputfile)
    if not videofile.isOpened():
        raise Exception("the video file can not be open")
    
    # RECPOINT = ((400, 100), (640, 400))
    RECPOINT = [(350, 100), (700, 400)]
    COUNT = videofile.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:
        # random choose the frame to demonstrate
        randframe = np.random.randint(COUNT)
        videofile.set(cv2.CAP_PROP_POS_FRAMES, randframe)
        ret, frame = videofile.read()
        if ret is False:
            break
        print('%d/%d' % (randframe, int(COUNT)), end='\r')
        # cv2.rectangle(frame, RECPOINT[0], RECPOINT[1], [0, 0, 255], 2)
        img = frame[RECPOINT[0][1]:RECPOINT[1][1], RECPOINT[0][0]:RECPOINT[1][0], :]
        # cv2.imwrite('%d.jpg' % randframe, img)
        MotionData_every_frame(deepcopy(img), mode, display=True)


def MotionData_every_frame(oriImg, mode='body', display=False):
    '''
    description: Extract the motion data from every frame data
    param:
        oriImg: the input frame data
        mode: the body or bodyhand, this can reference to 
            'Extract_MotionData_from_Video' function
        display:(bool) whether to display the extracted data on the frame
    return {array): the motion data of this frame image
    author: mario
    '''
    # first to estimate the body info
    candidate, subset = body_estimation(oriImg)
    PoseMat = np.zeros((60, 3))

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
    
    # clear the other person data
    for i in range(len(subset)):
        if i != maxindex:
            subset[i, :] = -1

    # detect and extract the hand
    if mode == 'bodyhand':
        # get the ROI of hands
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
            # the hand_estimation can only detect right hand, so the left hand should be flip (right-left) first
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
            all_hand_peaks.append(peaks[:, 0:2])
    
    # draw the body and hand 
    if display is True:
        canvas = deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        if mode == 'bodyhand':
            canvas = util.draw_handpose(canvas, all_hand_peaks)
            plt.imshow(canvas[:, :, [2, 1, 0]])
            plt.axis('off')
            plt.show()
        else:  # only body motion can displayed with opencv
            cv2.imshow('bodydata', canvas)
            cv2.waitKey(10)

    # return the motion data
    if mode != 'bodyhand':
        PoseMat = PoseMat[:18, :]
    
    return PoseMat


def CheckNpydata(videopath, datapath):
    '''
    description: verify the motiondata's correctness
    param: 
        videopath:(str) the path of the origin video
        datapath(str): the path of the extracted motin data
    return: None
    author: mario
    '''
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
        if not DisplayPose(img, pose, mode):
            break
        index += 1


def DisplayPose(img, pose, mode='body'):
    '''
    description: display the posedata in the img
    param:
        img: BGR image data
        posed: the joint position of the skeleton or hands
        mode: the type of the motion data
    return {type} 
    author: mario
    '''
    if mode == 'body':
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(pose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = pose[i, :2]
        canvas = util.draw_bodypose(img, candidate, subset)
        cv2.imshow('img', canvas)
        key = cv2.waitKey(30)
        if key & 0xff == ord('q'):
            return False
    else:
        pass
    return True
        

def CombineMotiondata(datafolder, mode):
    '''
    description: conbine the extract data into one dictinary file and save
    param: 
        datafolder(str): the saved motiondata file
        mode(str): 'body' or 'bodyhand'
    return: None
    author: mario
    '''
    MotionDataDicVideos = {}
    for i in range(1, 93):
        if mode == 'body':
            filename = 'e%d-bd.npy' % i
        elif mode == 'bodyhand':
            filename = 'e%d-hb.npy' % i
        filepath = os.path.join(datafolder, filename)
        if os.path.exists(filepath):
            motiondata = np.load(filepath)
            Postion = motiondata[:, :, :2].astype(np.int16)
            Score = motiondata[:, :, -1]
            key = '%dvideo' % i
            MotionDataDicVideos[key] = (Postion, Score)
    joblib.dump(MotionDataDicVideos, '../data/motionsdic.pkl')


def Verifypkldata(videodir, pklfilepath):
    motiondtadic = joblib.load(pklfilepath)
    for videoindex in range(93):
        videoname = 'e%d.avi' % videoindex
        videopath = os.path.join(videodir, videoname)
        # prepare the video and motiondata
        if not os.path.exists(videopath):
            print('the file %s is not exist' % videopath)
            continue
        video = cv2.VideoCapture(videopath)
        videokey = '%dvideo' % videoindex
        PoseMat = motiondtadic[videokey][0]  
        # determine the mode using the shape of data
        mode = 'bodyhand'
        if PoseMat.shape[1] == 18:
            mode = 'body'      

        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if count != PoseMat.shape[0]:
            print('the count number is not equal')
        randframs = np.random.randint(count, size=(1000,))

        Recpoint = [(350, 100), (700, 400)]
        for randframe in randframs:
            video.set(cv2.CAP_PROP_POS_FRAMES, randframe)
            ret, frame = video.read()
            img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
            if not DisplayPose(img, PoseMat[randframe, :, :], mode):
                break


def Demons_SL_video_clip(clipindex, poseclip):
    '''
    description: loop play a video clip
    param:
        clipindex: [videoindex, framebegin, frameend, ...]
        poseclip: the extracted pose motion clip
    return {type} 
    author: mario
    '''
    videodir = '/home/mario/sda/signdata/bbcpose'
    Recpoint = [(350, 100), (700, 400)]

    videoindex, beginindex, endindex = clipindex[:3]
    videoname = 'e%d.avi' % (videoindex+1)
    videopath = os.path.join(videodir, videoname)
    
    video = cv2.VideoCapture(videopath)
    index = 0
    video.set(cv2.CAP_PROP_POS_FRAMES, beginindex)
    while True:
        if index >= endindex-beginindex:
            index = 0
            video.set(cv2.CAP_PROP_POS_FRAMES, beginindex)
        
        ret, frame = video.read()
        img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
        if len(poseclip) == endindex - beginindex:
            if not DisplayPose(img, poseclip[index, :, :]):
                break
        else:
            cv2.imshow('img', img)
            key = cv2.waitKey(30)
            if key == ord('q'):
                break
        index += 1


if __name__ == "__main__":

    args = sys.argv[1:]
    begin = 0
    for arg in args:
        begin = int(arg)

    destfolder = '/home/mario/sda/signdata/bbcpose'
    # destfolder = '/home/hit605/public/Upload/mx/bbcpose'
    filenames = os.listdir(destfolder)

    Code = 4

    if Code == 1:  # random choose one file to estimation
        print('testcode1')
        filename = filenames[np.random.randint(len(filenames))]
        filepath = os.path.join(destfolder, filename)
        Demo_motion_of_Video(filepath)
    elif Code == 2:
        print('testcode2')
        filenames.sort()
        end = min(len(filenames), begin+10)
        for filename in filenames[begin:end]:
            filepath = os.path.join(destfolder, filename)
            name = filename.split('.')[0]
            print(name)
            Extract_MotionData_from_Video(filepath, name, mode='body')
    elif Code == 3:
        print('testcode3')
        filenames.sort()
        num = np.random.randint(len(filenames))
        for filename in filenames[num:]:
            filepath = os.path.join(destfolder, filename) 
            name = filename.split('.')[0]
            datapath = '../data/%s-bd.npy' % name
            if os.path.isfile(datapath):
                print(name)
                CheckNpydata(filepath, datapath)
    # combine the motion data files to one file
    elif Code == 4:
        npydatafolder = '/home/mario/sda/signdata/bbcpose_npy'
        print('testcode4')
        # CombineMotiondata(npydatafolder, 'body')
        if not os.path.exists('../data/motionsdic.pkl'):
            CombineMotiondata(npydatafolder, 'body')
        Verifypkldata(destfolder, '../data/motionsdic.pkl')