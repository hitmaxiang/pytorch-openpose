import cv2
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time 

from src import model
from src import util
from src.body import Body
from src.hand import Hand


# load the trained model of pose and hand 
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')


def EstimationVideo(inputfile):

    # open the video and read every frame data
    videofile = cv2.VideoCapture(inputfile)

    if not videofile.isOpened():
        raise Exception("the video file can not be open")
    
    RECPOINT = ((400, 100), (640, 400))
    COUNT = videofile.get(cv2.CAP_PROP_FRAME_COUNT)
    while True:
        randframe = np.random.randint(COUNT)
        videofile.set(cv2.CAP_PROP_POS_FRAMES, randframe)
        ret, frame = videofile.read()
        if ret is False:
            break
        # cv2.rectangle(frame, RECPOINT[0], RECPOINT[1], [0, 0, 255], 2)
        img = frame[RECPOINT[0][1]:RECPOINT[1][1], RECPOINT[0][0]:RECPOINT[1][0], :]
        # cv2.imwrite('%d.jpg' % randframe, img)
        '''
        cv2.imshow('frame', img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        '''
        print(randframe)
        # begin = time.time()
        EsimationFrame(deepcopy(img))
        # print(time.time()-begin)
    videofile.release()


def EsimationFrame(oriImg):
    candidate, subset = body_estimation(oriImg)
    canvas = deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # cv2.imshow('body', canvas)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    for x, y, w, is_left in hands_list:

        if is_left:
            cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x, y), (x+w, y+w), (255, 0, 0), 2, lineType=cv2.LINE_AA)
        
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        '''
        if is_left:
            plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            plt.show()
        '''
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
            
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
    
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    EstimationVideo('1.avi')
