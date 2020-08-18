import cv2
import os


def picvideo(path, size):
    filelist = os.listdir(path)
    filenum = [int(snum.split('.')[0]) for snum in filelist]
    filenum.sort()
    filelist = ['%d.jpg' % num for num in filenum]
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    VideoWriter = cv2.VideoWriter('a.avi', fourcc, fps, size)

    N = filenum[-1]
    M = N // 20
    i = 0
    for item in filelist:
        if i % M == 0:
            print('process %.2f %%' % (i/N*100))
        i += 1
        if item.endswith('.jpg'):
            imgfile = os.path.join(path, item)
            # print(imgfile)
            img = cv2.imread(imgfile)
            img = cv2.resize(img, size)
            VideoWriter.write(img)
    VideoWriter.release()
    print('success')


if __name__ == '__main__':
    path = '/home/mario/sda/sign data/bbcpose_data_1.0/1'
    img1 = os.path.join(path, '1.jpg')
    img = cv2.imread(img1)
    shape = img.shape
    size = (shape[1], shape[0])
    picvideo(path, size)