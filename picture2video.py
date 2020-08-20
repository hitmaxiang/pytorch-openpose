import cv2
import os
import numpy as np


def pictures2video(path, size, outputfile, fps=30):
    
    # convert the filename into num and sort
    filelist = os.listdir(path)
    filenum = [int(snum.split('.')[0]) for snum in filelist]
    filenum.sort(reverse=False)  # increase sequence
    filelist = ['%d.jpg' % num for num in filenum]

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
        print('process %.2f %%' % (i/N*100), end='\r')
        i += 1
        if item.endswith('.jpg'):
            imgfile = os.path.join(path, item)
            img = cv2.imread(imgfile)
            img = cv2.resize(img, size)
            VideoWriter.write(img)
    VideoWriter.release()
    VideoFrameCheck(path, filelist, outputfile)
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
def VideoFrameCheck(path, filelist, outputfile):
    videofile = cv2.VideoCapture(outputfile)
    N = videofile.get(cv2.CAP_PROP_FRAME_COUNT)
    # print('the (width, height) is (%d, %d)' % (videofile.get(3), videofile.get(4)))
    size = (int(videofile.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videofile.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if N != len(filelist):
        print('the frame number of video is not equal to the number of images')
    
    checktimes = 70
    randframes = np.random.randint(N, size=checktimes)
    errors = np.zeros((checktimes,))

    for index, frameindex in enumerate(randframes):
        videofile.set(cv2.CAP_PROP_POS_FRAMES, frameindex)
        ret, frameimg = videofile.read()
        image = cv2.imread(os.path.join(path, filelist[frameindex]))
        image = cv2.resize(image, size)
        errors[index] = np.sum(np.sum(np.sum((image-frameimg)**2)))**(0.5)
        # cv2.imshow('1', frameimg)
        # cv2.imshow('2', image)
        # cv2.imshow('1-2', abs(frameimg-image))
        # cv2.waitKey(0)
    cv2.destroyAllWindows()
    videofile.release()
    print("the error is %f" % (np.mean(errors)/(size[0]*size[1])))


if __name__ == '__main__':
    path = '/home/mario/sda/sign data/bbcpose_data_1.0/1'
    img1 = os.path.join(path, '1.jpg')
    img = cv2.imread(img1)
    shape = img.shape
    size = ((shape[1]//2)*2, (shape[0]//2)*2)
    pictures2video(path, size, '1.avi')