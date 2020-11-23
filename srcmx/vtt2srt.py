import os
import re
import cv2
import joblib
import numpy as np

from scipy.io import loadmat


def seconds2timestring(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def vtt2srt(inputfile, outputpath):
    '''
    description: 将vtt的字幕文件转变为srt文件
    param： vtt 字幕文件的路径
    return {*}
    author: mario
    '''
    # 读取 vtt 字幕文件中的所有文件 
    with open(inputfile, 'r') as f:
        lines = f.readlines()

    # 初始化待写入的字幕的时间戳以及字符串
    previous_sub = ''
    previous_time = ''

    # 定义时间戳和字幕字符串的 re 模式
    # 时间戳实例： 00:00:00.640 --> 00:00:02.869 align:start position:0%
    # 字符串实例 good<00:00:00.880><c> afternoon</c><00:00:01.520><c> and</c>
    # 因为在 vtt 格式的字幕中， 存在动态字幕， 所以需要从中提取
    time_pattern = r'(\d{2}:\d{2}:\d{2}).(\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}).(\d{3})'
    dyn_sub_pattern = r'<\d\d:\d\d:\d\d.\d{3}><c>\s*([^<>]+)</c>'

    # create the output file for writing
    outputfile = open(outputpath, 'w')

    # intialize the count information and write_option_option
    # the write_option_op indicates that the previous reserved subtitle is write_option or not
    count = 1
    write_option = True
    last_end_time = 0

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        # extract the time stamp information
        args = re.findall(time_pattern, line)
        if len(args) != 0:
            args = args[0]
            begin_time = '%s,%s' % (args[0], args[1])
            end_time = '%s,%s' % (args[2], args[3])
            current_time = [begin_time, end_time]

            # when the reserved subtitle info is not write, then the endtime of 
            # the previous_sub should be update to last_end_time
            if write_option is False and previous_time != '':
                previous_time[1] = last_end_time
            
            # initialzie the first time stamp
            if previous_time == '':
                previous_time = current_time
            
            # reset the write_option to false
            write_option = False

        # when the line is string (after the first time stamp is read)
        elif previous_time != '':
            # extract the words using the split with the dynamic suntitle pattern 
            words = re.split(dyn_sub_pattern, line)
            sentence = ''
            for word in words:
                if word != '':
                    sentence += ' %s' % word
            sentence = sentence.strip()
            
            # when the current string is not same as the previous subtitle, the previous suntitle should be write
            if sentence != previous_sub and previous_sub != '':
                outputfile.write_option('%d\n%s --> %s\n%s\n\n' % (count, previous_time[0],
                                        previous_time[1], previous_sub))

                # after the write, the states of the previous should be update to current states
                previous_time = current_time
                count += 1
                write_option = True
            last_end_time = current_time[1]
            previous_sub = sentence

    outputfile.close()


def srt2txt(inputfile, outfilepath, fps):
    with open(inputfile) as f:
        lines = f.readlines()
    
    time_pattern = r'(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    count_pattern = r'^\d+$'

    multiper = np.array([3600, 60, 1, 1e-3]) * fps

    count_ref = 1

    outfile = open(outfilepath, 'w')
    for line in lines:
        if line.strip() == '':
            continue

        count = re.findall(count_pattern, line)
        if len(count) != 0:
            if int(count[0]) == count_ref:
                count_ref += 1
                continue
        
        stamp = re.findall(time_pattern, line)
        
        if len(stamp) != 0:
            framecount = [0, 0]
            for index in range(len(stamp)):
                t_stamp = np.array([int(s) for s in stamp[index]])
                framecount[index] = int(np.dot(t_stamp, multiper))
            continue
        
        outfile.write('%8d\t%8d\t%s' % (framecount[0], framecount[1], line))
    outfile.close()


def txt2dict(videodir, subdir):
    SubtitleDict = {}
    # extract the subtitle for all the giving video
    videonames = os.listdir(videodir)
    for videoname in videonames:
        # make sure it is a video file
        name, ext = os.path.splitext(videoname)
        if ext in ['.mp4', '.avi']:
            # extract the key word of the dictionary
            key = name[:3]
            # find the corresponding subtitle file
            subpath = os.path.join(subdir, name+'.en.txt')
            if os.path.isfile(subpath):
                subdata = []

                with open(subpath) as f:
                    lines = f.readlines()

                for line in lines:
                    begin, end, text = line.split('\t')
                    begin, end = int(begin), int(end)
                    text = text.strip()

                    if len(subdata) != 0 and len(text.split()) <= 3 and (begin-subdata[-1][1]) <= 2:
                        subdata[-1][1] = end
                        subdata[-1][-1] += (' ' + text)
                    else:
                        subdata.append([begin, end, text])
                
                SubtitleDict[key] = subdata
    
    joblib.dump(SubtitleDict, 'SubtitleDict.pkl')
    
    return SubtitleDict


def mat2dict(matfile, outname, keystring='bbc_subtitles'):
    
    # extract the data from the .mat file, the data format in the mat file is:
    # 1. the data is stored in dict with keystring like: "bbc_subtitles"
    # 2. the data's format is 1xN array
    # 3. every array is [[videoname], [[[beginframe]],[[endframe]], [subtitles]]
    SubtitleDict = {}
    mat = loadmat(matfile)
    # get the N items
    datas = mat[keystring][0]
    for item in datas:
        videoname = item[0][0]
        records = item[1][0]
        subdata = []
        for record in records:
            begin = record[0][0]
            end = record[1][0]
            text = record[2][0].strip()
            if len(subdata) != 0 and len(text.split()) <= 3 and (begin-subdata[-1][1]) <= 2:
                subdata[-1][1] = end
                subdata[-1][-1] += (' ' + text)
            else:
                subdata.append([begin, end, text])
        SubtitleDict[videoname] = subdata
    joblib.dump(SubtitleDict, outname)


if __name__ == "__main__":
    TestCode = 2
    
    # subtitle file dir
    vttsubdir = '/home/mario/sda/signdata/SPBSL/subtitle/vtt'
    srtsubdir = '/home/mario/sda/signdata/SPBSL/subtitle/srt'
    txtsubdir = '/home/mario/sda/signdata/SPBSL/subtitle/txt'
    
    # video directory
    normal_videodir = '/home/mario/sda/signdata/SPBSL/scenes/normal/video'

    if TestCode == 0:
        # convert the vtt file to srt file and txt file
        vttnames = os.listdir(vttsubdir)
        for vttname in vttnames:
            filename, ext = os.path.splitext(vttname)
            vttfilepath = os.path.join(vttsubdir, vttname)
            srtfilepath = os.path.join(srtsubdir, filename+'.srt')
            if ext == '.vtt':
                vtt2srt(vttfilepath, srtfilepath)
    
    elif TestCode == 1:
        # convert the srt file 2 txtfile
        infofile = open('info.txt', 'w')
        videonames = os.listdir(normal_videodir)
        for videoname in videonames:
            name, ext = os.path.splitext(videoname)
            videopath = os.path.join(normal_videodir, videoname)
            srtpath = os.path.join(srtsubdir, name+'.en.srt')
            txtpath = os.path.join(txtsubdir, name+'.en.txt')
            if ext == '.mp4':
                if os.path.exists(videopath) and os.path.exists(srtpath):
                    video = cv2.VideoCapture(videopath)
                    if video.isOpened():
                        fps = video.get(cv2.CAP_PROP_FPS)
                        counts = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        infos = '%s---%f---%d--%s' % (name[:3], fps, counts, seconds2timestring(counts/fps))
                        infofile.write('%s\n' % infos)
                        srt2txt(srtpath, txtpath, fps)
                    video.release()
        infofile.close()

    elif TestCode == 2:
        # construct the subtitle dictionary based on the txt file
        subtitledict = txt2dict(normal_videodir, txtsubdir)
        
        datatxt = open('subdictxt.txt', 'w')
        # check the dictionary data
        for key in subtitledict.keys():
            datatxt.write('\n\nvideo-%s\n\n' % key)
            datas = subtitledict[key]
            for begin, end, text in datas:
                datatxt.write('%8d\t%8d\t%s\n' % (begin, end, text))
        datatxt.close()