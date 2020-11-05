import sys
sys.path.append('..')
import cv2
import os
import time
import joblib
import utilmx
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

# the recpoint of the bbc news
gl_RECPOINT = [(350, 100), (700, 400)]


def Extract_MotionData_from_Video(videopath, outpath, Recpoint, mode='body', overwrite=False):
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
        # outname = '../data/%s-hb' % outname
    else:
        MotionMat = np.zeros((count, 18, 3))
        # outname = '../data/%s-bd' % outname
    
    # the user-defined ROI
    # Recpoint = gl_RECPOINT

    # npyname = '%s.npy' % outname

    # if os.path.isfile(npyname) and overwrite is False:
        # return
    outname = os.path.split(outpath)[1]
    index = 0
    while True:
        ret, frame = video.read()
        # if ret is False or index > 200:
        if ret is False:
            break
        # print('%s-%d/%d' % (outname, index, int(count)))
        # print the progress information every 100 frame
        if index % 100 == 0:
            print('%s-%d/%d' % (outname, index, int(count)))
        # crop the ROI of the frame
        img = frame[Recpoint[0][1]:Recpoint[1][1], Recpoint[0][0]:Recpoint[1][0], :]
        PoseMat = MotionData_every_frame(img, mode, display=False)
        MotionMat[index, :, :] = PoseMat
        index += 1
    joblib.dump(MotionMat, outpath)
    print('%s is saved!' % outpath)


def Demo_motion_of_Video(inputfile, mode, recpoint, random=True, Maxiter=None):
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
    RECPOINT = recpoint
    COUNT = int(videofile.get(cv2.CAP_PROP_FRAME_COUNT))

    if Maxiter is None:
        Maxiter = COUNT
    else:
        Maxiter = min(COUNT, Maxiter)

    Iterations = 0

    while True:
        # random choose the frame to demonstrate
        if random is True:
            videofile.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(COUNT))
        ret, frame = videofile.read()
        Iterations += 1
        if ret is False or Iterations >= Maxiter:
            break

        print('%20s' % ' ', end='\r')
        print('%d/%d/%d' % (Iterations, Maxiter, COUNT), end='\r')
        # cv2.rectangle(frame, RECPOINT[0], RECPOINT[1], [0, 0, 255], 2)
        img = frame[RECPOINT[0][1]:RECPOINT[1][1], RECPOINT[0][0]:RECPOINT[1][0], :]
        # cv2.imwrite('%d.jpg' % randframe, img)
        MotionData_every_frame(deepcopy(img), mode, display=True)
        
        key = cv2.waitKey(5) & 0xff
        if key == ord('q'):
            break


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
    begin_time = time.time()
    candidate, subset = body_estimation(oriImg)
    print('each const time %f seconds' % (time.time()-begin_time))
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
            canvas = utilmx.draw_handpose_by_opencv(canvas, all_hand_peaks)
            cv2.imshow('bodyhand', canvas)
            # canvas = util.draw_handpose(canvas, all_hand_peaks)
            # plt.imshow(canvas[:, :, [2, 1, 0]])
            # plt.axis('off')
            # plt.show()
        else:  # only body motion can displayed with opencv
            cv2.imshow('bodydata', canvas)
            # cv2.waitKey(10)

    # return the motion data
    if mode != 'bodyhand':
        PoseMat = PoseMat[:18, :]
    
    return PoseMat


def Checkout_motion_data(videopath, datapath, recpoint):
    '''
    description: verify the motiondata's correctness
    param: 
        videopath:(str) the path of the origin video
        datapath(str): the path of the extracted motin data
    return: None
    author: mario
    '''
    video = cv2.VideoCapture(videopath)
    if datapath.endswith('npy'):
        PoseMat = np.load(datapath)
    elif datapath.endswith('pkl'):
        PoseMat = joblib.load(datapath)
    
    if not video.isOpened():
        print('the file %s is not exist' % videopath)
        return
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if count != PoseMat.shape[0]:
        print('the count number is not equal')
    
    # mincount = min(count, PoseMat.shape[0])

    mode = 'bodyhand'
    if PoseMat.shape[1] == 18:
        mode = 'body'

    index = 0
    while True:
        ret, frame = video.read()
        if ret is False:
            break
        print('%d/%d' % (index, int(count)), end='\r')
        img = frame[recpoint[0][1]:recpoint[1][1], recpoint[0][0]:recpoint[1][0], :]
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
    if mode == 'body' or mode == 'bodyhand':
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(pose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = pose[i, :2]
        canvas = util.draw_bodypose(img, candidate, subset)
        
        if mode == 'bodyhand':
            peaks = pose[18:, :2]
            peaks = np.reshape(peaks, (2, -1, 2))
            canvas = utilmx.draw_handpose_by_opencv(canvas, peaks)

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

        Recpoint = gl_RECPOINT
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
    Recpoint = gl_RECPOINT

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


def Demo_original_video(inputfile, random=False, Recpoint=None, Maxiter=None):
    video = cv2.VideoCapture(inputfile)
    if video.isOpened():
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if Recpoint is None:
            Recpoint = [(0, 0), (width, height)]
        Iterations = 0
        while True:
            ret, frame = video.read()
            Iterations += 1
            if ret is False or (Maxiter is not None and Iterations >= Maxiter):
                break
            
            if random is True:
                video.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(framecount))
                        
            cv2.rectangle(frame, Recpoint[0], Recpoint[1], color=[0, 0, 255], thickness=2)
            cv2.imshow('demo-video', frame)

            key = cv2.waitKey(10) & 0xff
            if key == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()


def Test(Code, init=False, mode='body', dataset='spbsl', server=False):

    # initialize the data source according the input option
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

    if Code == 0:
        # demonstrate the original video file
        print('play the video and verify the ROI rectangle')
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                Demo_original_video(filepath, random=True, Recpoint=Recpoint)
    
    elif Code == 1:
        # random choose one file to estimation
        print('random choose files to demonstrate the motion extraction process')
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                print(filename)
                # Dem_video(filepath, Recpoint=Recpoint)
                Demo_motion_of_Video(filepath, mode, recpoint=Recpoint, Maxiter=200)
    
    elif Code == 2:
        print('extract the motion data from the videos and save')
        np.random.shuffle(filenames)
        for filename in filenames:
            complet_files = utilmx.Records_Read_Write().Get_extract_ed_ing_files(datadir, init)
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                outname = 'video-%s-%s.pkl' % (filename[:3], mode)
                if outname not in complet_files:
                    print(outname)
                    utilmx.Records_Read_Write().Add_extract_ed_ing_files(datadir, outname)
                    outpath = os.path.join(datadir, outname)
                    Extract_MotionData_from_Video(filepath, outpath, Recpoint, mode)
                    # break

    elif Code == 3:
        # check the correctness of the extracted motion data
        print('check the correctness of the extracted motion data')
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                filepath = os.path.join(videofolder, filename)
                outname = 'video-%s-%s.pkl' % (filename[:3], mode)
                outpath = os.path.join(datadir, outname)
                if os.path.isfile(outpath):
                    print(outname)
                    Checkout_motion_data(filepath, outpath, Recpoint)
    # combine the motion data files to one file
    elif Code == 4:
        npydatafolder = '/home/mario/sda/signdata/bbcpose_npy'
        print('testcode4')
        # CombineMotiondata(npydatafolder, 'body')
        if not os.path.exists('../data/motionsdic.pkl'):
            CombineMotiondata(npydatafolder, 'body')
        Verifypkldata(destfolder, '../data/motionsdic.pkl')


if __name__ == "__main__":
    argv = sys.argv[1:]
    # Code, init=False, mode='body', dataset='spbsl', server=False
    # print(argv)
    if len(argv) == 5:
        Code = int(argv[0])
        init = [False, True][int(argv[1])]
        mode = ['body', 'bodyhand'][int(argv[2])]
        dataset = ['bbc', 'spbsl'][int(argv[3])]
        server = [False, True][int(argv[4])]
        print(Code, init, mode, dataset, server)
        if init is False:
            time.sleep(20)
        Test(Code, init, mode, dataset, server)
    else:
        Test(3, mode='body', server=True)