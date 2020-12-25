'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-12-25 17:55:59
LastEditors: mario
LastEditTime: 2020-12-25 22:53:57
'''

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt 


def Get3Dthumbnails(videopath, frames, outname, fx=1, rec=None):
    if os.path.exists(videopath):
        video = cv2.VideoCapture(videopath)
        Count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        indexes = np.random.randint(Count, size=(frames,))
        images = []
        for i in range(frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, indexes[i])
            _, img = video.read()
            img = cv2.resize(img, (0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            if rec is not None:
                img = img[rec[0][1]:rec[1][1], rec[0][0]:rec[1][0], :]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转化为 RGB 颜色空间
                img = cv2.flip(img, 0)  # 垂直翻转
            images.append(img)
        images = np.array(images)
        video.release()
    else:
        images = np.random.rand(frames, 320, 160, 3)
    
    N, H, W = images.shape[:3]
    x = np.arange(0, W)
    z = np.arange(0, H)

    X, Z = np.meshgrid(x, z)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')

    Y = np.zeros_like(X)

    offset = 8
    for i in range(frames):
        # 归一化转化为【0-1】颜色
        colormat = np.reshape(images[i], (-1, 3))/255.0
        # 
        stride = offset * i
        ax.scatter(X + stride, Y + stride, Z + stride, c=colormat, s=0.5)

    ax.grid(None)
    print('elev %f' % ax.elev)
    print('azim %f' % ax.azim)
    # elev 表示 z 的 rotation，默认是向下低头 30（30）
    # zim 表示 xy 面的 rotation， 默然是向 左旋转 30 （即-60），
    ax.view_init(elev=15, azim=-75)
    plt.axis('off')
    plt.savefig(outname)
    plt.show()


def RunTest(testcode, server):
    if server:
        videodir = '/home/mario/signdata/spbsl/normal'
    else:
        videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'
    
    Recpoint = [(700, 100), (1280, 720)]
    
    if testcode == 0:
        filenames = os.listdir(videodir)
        for filename in filenames:
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                videopath = os.path.join(videodir, filename)
                Get3Dthumbnails(videopath, 25, '../data/img/videothumbnail.jpg', rec=Recpoint)
                break  # only onetime
    

if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('-t', '--testcode', type=int, default=0)
    Parser.add_argument('-s', '--server', action='store_true')

    args = Parser.parse_args()
    testcode = args.testcode
    server = args.server
    
    RunTest(testcode, server)