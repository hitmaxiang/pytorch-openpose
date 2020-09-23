import cv2
import os
import numpy as np
import multiprocessing
from numba import jit
import utilmx


def GetorderFilelist(path):
    filelist = os.listdir(path)
    filenum = [int(snum.split('.')[0]) for snum in filelist]
    filenum.sort(reverse=False)  # increase sequence
    filelist = ['%d.jpg' % num for num in filenum]
    return filenum, filelist


# @jit
def pictures2video(path, outputfile, fps=30):
    
    # convert the filename into num and sort
    filenum, filelist = GetorderFilelist(path)

    Startfile = os.path.join(path, filelist[0])
    img = cv2.imread(Startfile)
    shape = img.shape
    size = ((shape[1]//2)*2, (shape[0]//2)*2)
    # check the time continuity of the filelist
    TimeContinuityCheck(filelist)
    # create the video writer , and saved as outfile
    if not outputfile.endswith('.avi'):
        print('the video will be saved as .avi type')
        outputfile = outputfile.split('.')+'.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoWriter = cv2.VideoWriter(outputfile, fourcc, fps, size)

    # set the process print info variable
    N = len(filelist)
    i = 0
    for item in filelist:
        if i % 10 == 0:
            print('process %.2f %%' % (i/N*100), end='\r')
        i += 1
        if item.endswith('.jpg'):
            imgfile = os.path.join(path, item)
            img = cv2.imread(imgfile)
            img = cv2.resize(img, size)
            VideoWriter.write(img)
    VideoWriter.release()
    with open('bbcposestartinfo.txt', 'a') as f:
        f.write('the start index of %s is %d\n' % (os.path.basename(path), filenum[0]))
    print('success')


def TimeContinuityCheck(filelist):
    Length = len(filelist)
    Minnum = int(filelist[0].split('.')[0])
    # Maxnum = int(filelist[-1].split('.')[0])

    for i in range(Length):
        if i != (int(filelist[i].split('.')[0])-Minnum):
            raise Exception('the time continuity error occur at %s' % filelist[i])
    print('the time continuity checked')


# check the random choosed frame to check the file 
def VideoFrameCheck(srcfolder, destfolder, index, checktime=50):

    for i in index:
        path = os.path.join(srcfolder, str(i))
        videofile = '%s/e%d.avi' % (destfolder, i)
        
        if os.path.isfile(videofile) and os.path.isdir(path):

            filenum, filelist = GetorderFilelist(path)
            video = cv2.VideoCapture(videofile)
            N = video.get(cv2.CAP_PROP_FRAME_COUNT)
            # print('the (width, height) is (%d, %d)' % (video.get(3), video.get(4)))
            size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if N != len(filelist):
                print('the frame number of video is not equal to the number of images')
            
            randframes = np.random.randint(N, size=checktime)

            for index, frameindex in enumerate(randframes):
                video.set(cv2.CAP_PROP_POS_FRAMES, frameindex)
                ret, frameimg = video.read()
                image = cv2.imread(os.path.join(path, filelist[frameindex]))
                image = cv2.resize(image, size)
                diffimg = cv2.absdiff(frameimg, image)
                
                # errors = np.sum(np.sum(np.sum(diffimg**2)))/(size[0]*size[1])
                errors = np.max(np.max(np.sum(diffimg**2, axis=-1)))
                errors = errors**(0.5)
                if errors > 26:
                    print('%d-%d/%d--errors: %f' % (i, index, checktime, errors))
                    # print('too greate')
                cv2.imshow('img', frameimg)
                cv2.imshow('video', image)
                cv2.imshow('diff', diffimg)
                cv2.imshow('add', cv2.addWeighted(frameimg, 0.5, image, 0.5, 0))
                key = cv2.waitKey(0)
                if key == 'q':
                    break
            video.release()
    cv2.destroyAllWindows()


def Img2VideoBatch(srcfolder, destfolder, index):
    P = multiprocessing.Pool(processes=4)
    for i in index:
        print('start to transfer video %d' % i)
        path = '%s/%d' % (srcfolder, i)
        destfile = '%s/e%d.avi' % (destfolder, i)
        if os.path.isdir(path) and os.path.isdir(destfolder):
            P.apply_async(pictures2video, args=(path, destfile, 30))
            # pictures2video(path, destfile)
        else:
            print('the file or folder is not exist!')
    P.close()
    P.join()


def UnzipImages(zipfolder, destfolder, index):
    os.chdir(destfolder)
    for i in index:
        filename = '%s/%d.tgz' % (destfolder, i)
        if os.path.isfile(filename):
            os.system('tar -xvf %s' % filename)
            print('the %d.tgz is unziped' % i)


def CheckMotionRegion(videofolder):
    videofiles = os.listdir(videofolder)
    videofiles.sort()
    RecBorder = [(350, 100), (700, 400)]
    # FaceCenter = (520-RecBorder[0][0], 160-RecBorder[0][1])
    # RecBorder = ((400, 100), (640, 400))
    for videoname in videofiles:
        videopath = os.path.join(videofolder, videoname)
        video = cv2.VideoCapture(videopath)
        w, h = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        Count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        if video.isOpened():
            print('the shape of %s is (%d, %d)' % (videoname, w, h))
            cv2.namedWindow(videoname.split('.')[0])
        else:
            print('the file %s can not be opened!' % videoname)
            video.release()
            continue

        while True:
            frameindex = np.random.randint(Count)
            video.set(cv2.CAP_PROP_POS_FRAMES, frameindex)
            ret, frame = video.read()
            if ret is True:
                cv2.rectangle(frame, RecBorder[0], RecBorder[1], [0, 0, 255], 2)
                rec = frame[RecBorder[0][1]:RecBorder[1][1], RecBorder[0][0]:RecBorder[1][0], :]
                Faces, frame = utilmx.FaceDetect(rec)
                cv2.imshow(videoname.split('.')[0], frame)
                key = cv2.waitKey(0)
                if key & 0xff == ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    # srcfolder = '/media/mario/maxiang/bbcpose_extbbcpose_data_1.0'
    srcfolder = '/usr/bbcpictures'
    zipfolder = '/home/mario/sda/signdata/bbcpose_data_1.0'
    destfolder = '/home/mario/sda/signdata/bbcpose'
    Code = 5

    # batch conbine image sequences to video
    if Code == 1:
        # index = sys.argv[1:]
        index = [i for i in range(88, 89)]
        if len(index) != 0:
            index = [int(s) for s in index]
            Img2VideoBatch(srcfolder, destfolder, index)
            VideoFrameCheck(srcfolder, destfolder, index, 200)
            print(index)
        else:
            print('please input the index argument')
    
    # check the content of between images and video frames
    elif Code == 2:
        index = [i for i in range(1, 21)]
        VideoFrameCheck(srcfolder, destfolder, index, 100)

    # unzip the compressed images
    elif Code == 3:
        index = [i for i in range(1, 21)]
        UnzipImages(zipfolder, srcfolder, index)
    
    elif Code == 4:
        CheckMotionRegion(destfolder)
    
    elif Code == 5:
        utilmx.Getvideoframes(destfolder)
