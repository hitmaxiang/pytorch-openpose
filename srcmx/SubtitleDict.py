'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-09-15 14:46:38
LastEditors: mario
LastEditTime: 2020-10-17 15:44:27
'''
import os
import re
import time
import random
import joblib
import numpy as np
from scipy.io import loadmat
from copy import deepcopy

from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
Porter_Stemmer = PorterStemmer()
Lancas_Stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


class SubtitleDict():
    def __init__(self, dictfile, matfile=None, overwrite=False, midfile=False):
        '''
        description: initialize the subtitleDict class, when there is no
        args is giving, there is nothing to do
        param:
            dictfile: the filepath of the dict, if it is not exist, the matfile is need to construct
            matfile: the matlab format file (store the subtitles data)
            overwrite: whether to overwrite the file
        return: None 
        author: mario
        '''
        if (overwrite is True) or (not os.path.exists(dictfile)):
            if os.path.exists(matfile) and matfile.endswith('mat'):
                self.subtitledict = self.Construct_from_Mat(matfile, midfile)
                self.save(dictfile)
            else:
                print('the .mat file is needed')
        elif os.path.exists(dictfile) and dictfile.endswith('pkl'):
            self.subtitledict = joblib.load(dictfile)
        else:
            print('the .pkl dictfile path is needed')
        
        # get the maximum framcount of the videos
        self.FrameCounts = self.GetFrameCounts()
        
    def Construct_from_Mat(self, inputfile, midfile=False):
        '''
        description: extract the mat data into subtitledict
        param: 
            inputfile: str, the path of the .mat file 
        return: dict, the subtitledict 
        author: mario
        '''
        # extract the data from the .mat file, the data format in the mat file is:
        # 1. the data is stored in dict with key = "bbc_subtitles"
        # 2. the data's format is 1x92 array
        # 3. every array is [[name], [[[beginframe]],[[endframe]], [subtitles]]
        subtitledata = []
        mat = loadmat(inputfile)
        datas = mat['bbc_subtitles']
    
        for data in datas[0]:
            videoname = data[0][0]
            temprecord = [videoname]
            for record in data[1][0]:
                beginframe = record[0][0][0]
                endframe = record[1][0][0]
                subtile = record[2][0]
                temprecord.append([beginframe, endframe, subtile])
            subtitledata.append(temprecord)
        
        # using the subtitledata and engdict to construct subtitledict
        SubtitleDict = self.construct_subtiles_dict(subtitledata, midfile)

        # whether to save the midfile
        if midfile:
            # save the data of subtitledata in file
            with open('../midfile/matfile.mid', 'w') as f:
                for index, video in enumerate(subtitledata):
                    videoname = video[0]
                    f.write('\n\n\n===== e%d.avi ----- %s  ===\n' % (index+1, videoname))
                    videodata = video[1:]
                    for record in videodata:
                        f.write('%d\t%d\t%s\n' % (record[0], record[1], record[2]))

        return SubtitleDict
    
    def construct_subtiles_dict(self, subtitledata, midfile=False):
        '''
        description: split the words in subtitledata, and store every unique word
            (Remove grammatical tenses and singular and plural changes) in dict with 
            their positions [video, beginframe, endframe]
        param:
            subtitieldata: the subtitlelist extract from mat file
        return: 
            the subtitledictionaty
        author: mario
        '''

        # used to store the words that can't query in the dictionary
        SubtitleDict = {}
        for index, video in enumerate(subtitledata):
            # ignore the video without subtitles
            if len(video) == 1:
                continue
            videodata = video[1:]
            for jndex, record in enumerate(videodata):
                frame_diff = record[1] - record[0]
                exbegin, exend = record[0], record[1]

                # when the previous or back subtitlefram is close, it should be also consider 
                # as the candidate of the words of current subtitle
                if jndex >= 1:
                    if videodata[jndex-1][1] + frame_diff > record[0]:
                        exbegin = videodata[jndex-1][0]
                    
                if jndex <= len(videodata)-2:
                    if record[1] + frame_diff > videodata[jndex+1][0]:
                        exend = videodata[jndex+1][1]
                
                # deal with the sentence string
                words = self.Split_Sentence2words(record[2], mode=3)
                for word in words:
                    if word in SubtitleDict.keys():
                        SubtitleDict[word].append([index, exbegin, exend, record[0], record[1]])
                    else:
                        SubtitleDict[word] = [[index, exbegin, exend, record[0], record[1]]]
                        
        if midfile:
            # save the subtitledict word
            tempdictlist = sorted(SubtitleDict.items(), key=lambda x: len(x[1]))
            with open('../midfile/dictfile.mid', 'w', encoding='utf-8') as f:
                for word in tempdictlist:
                    f.write('%s\t%d\n' % (word[0], len(word[1])))
            
        return SubtitleDict
    
    # Split sentence to words
    def Split_Sentence2words(self, string, mode=0):
        '''
        description: split the sentence string into words
        param:
            string: str, the sentence string
            mode: the approach that used to split the senteces
                0: only split the sentence to words, and returen the original word
                1: using the porterstemmer to stem the word
                2: using the lancasterstammer to stem the word
                3: using the wordnet lemmatization to lemmatizer the word
        return: list, list of splited words
        author: mario
        '''
        string = string.lower()
        pattern = r"\b\w+'?\w*\b"
        tokens = re.findall(pattern, string)

        if mode == 0:
            words = tokens
        elif mode == 1:
            words = [Porter_Stemmer.stem(word) for word in tokens]
        elif mode == 2:
            words = [Lancas_Stemmer.stem(word) for word in tokens]
        elif mode == 3:
            word_tags = [(word, self.get_wordnet_pos(tag)) for word, tag in pos_tag(tokens)]
            words = [wordnet_lemmatizer.lemmatize(word, pos=tag) for word, tag in word_tags]
        # print(string, '\n', words)
        return words
    
    def get_wordnet_pos(self, treebank_tag):
        '''
        description: change the treebank pos-tag to wordnet's pos-tag
        param: treebank_tag, the treebank style pos_tag
        return: the worknet style pos-tag
        author: mario
        '''
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:  # the default pos-tag is set to noun
            return wordnet.NOUN
    
    def save(self, outpath):
        joblib.dump(self.subtitledict, outpath)

    def ChooseSamples(self, word, extend=True):
        begin_time = time.time()
        samples = self.subtitledict[word]
        neg_samples = []
        neg_keys = list(self.subtitledict.keys())
        # np.random.seed(25)
        while len(neg_samples) < len(samples):
            neg_key = np.random.choice(neg_keys)
            if neg_key == word:
                continue
            rindex = np.random.randint(len(self.subtitledict[neg_key])) 
            neg_sample = self.subtitledict[neg_key][rindex]
            Valid = True

            # when the endfram is large than the framecount of the video, then it will be ignore
            videoindex, beginindex, endindex = neg_sample[:3]
            if endindex > self.FrameCounts[videoindex] or (endindex-beginindex) < 40:
                continue
            for pos_sample in samples:
                if neg_sample[0] == pos_sample[0]:
                    if extend:
                        if not (neg_sample[2] <= pos_sample[1] or neg_sample[1] >= pos_sample[2]):
                            Valid = False
                            break
                    else:
                        if not (neg_sample[4] <= pos_sample[3] or neg_sample[3] >= pos_sample[4]):
                            Valid = False
                            break
            if Valid is True:
                neg_samples.append(neg_sample)
        print('the %s has %d samples, and consume %f seconds!' % (word, len(samples), time.time()-begin_time))
        return samples, neg_samples
    
    def GetFrameCounts(self):
        filepath = '../data/videoframeinfo.txt'
        FrameCounts = np.zeros((92, ), dtype=np.int32)
        pattern = r'^e(\d+).avi'
        with open(filepath) as f:
            lines = f.readlines()
        for line in lines:
            videoindex = int(re.findall(pattern, line)[0])
            framecount = int(line.split()[2])
            FrameCounts[videoindex-1] = framecount
        return FrameCounts


class AnnotationDict:
    def __init__(self, datapath):
        if os.path.isfile(datapath):
            if datapath.endswith('.mat'):
                self.annotation_dict = self.extract_annotationdict(datapath)
            elif datapath.endswith('.pkl'):
                self.annotation_dict = joblib.load(datapath)
    
    def extract_annotationdict(self, matpath):
        annotationdict = {}
        mat = loadmat(matpath)
        datas = mat['bbc_sign_annotation'][0]
        for index, data in enumerate(datas):
            if len(data[1][0]) == 0:
                continue
            else:
                for record in data[1][0]:
                    key = record[0][0]
                    begin_index = record[1][0][0]
                    end_index = record[2][0][0]
                    if key in annotationdict.keys():
                        annotationdict[key].append([index, begin_index, end_index])
                    else:
                        annotationdict[key] = [[index, begin_index, end_index]]
        joblib.dump(annotationdict, '../data/annotationdict.pkl')
        return annotationdict
    
    def Retrive_distance(self, word, indexes, locs):
        '''
        description: 根据对 Word 找到的位置计算 retrieve 与真实的差距
        param: 
            word: 搜索的关键词
            indexes: 候选 sub-sequences 的索引位置
            locs: 结果在 sub-sequences 的定位
        return: 
        author: mario
        '''
        if word not in self.annotation_dict.keys():
            print('the word in not in the annotation dictionary')
            return None
        # 获取真实的 word 的所在位置
        Records = self.annotation_dict[word]
        
        # 计算真实位置 在 retrieve 找寻过程中的搜寻结果
        rangedis = [float('inf')] * len(Records)
        for index, record in enumerate(Records):
            r_index, r_begin, r_end = record

            for i, q_loc in enumerate(locs):
                # get the retrieve locs and indexs 
                q_index, q_base = indexes[i]
                q_begin, q_end = q_loc + q_base
                
                # calculate the nearest distance 
                if q_index == r_index:
                    if r_begin <= q_begin <= r_end or r_begin <= q_end <= r_end:
                        rangedis[index] = 0
                    elif q_begin <= r_begin <= q_end or q_begin <= r_end <= q_end:
                        rangedis[index] = 0
                    else:
                        dis = min(np.abs([q_begin-r_begin, q_begin-r_end, q_end-r_begin, q_end-r_end]))
                        if dis < rangedis[index]:
                            rangedis[index] = dis
        return rangedis
        

def Test(testcode):
    subtilematpath = '../data/bbc_subtitles.mat'
    annotationpath = '../data/bbc_sign_annotation.mat'
    # test the subtiledict class
    if testcode == 0:
        # subdict = SubtitleDict(subtilefile, dictfile, midfile=True)
        subdictpath = '../data/subtitledict.pkl'
        subdict = SubtitleDict(dictfile=subdictpath, matfile=subtilematpath,
                               overwrite=True, midfile=True)
        subdict.ChooseSamples('snow')
    elif testcode == 1:
        anotationdict = AnnotationDict(annotationpath)



if __name__ == "__main__":
    Test(1)