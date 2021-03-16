'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-09-15 14:46:38
LastEditors: mario
LastEditTime: 2021-03-16 15:01:13
'''
import os
import re
import time
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from copy import deepcopy
from itertools import chain

from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords


class WordsDict():
    def __init__(self, worddictpath, subdictpath, overwrite=False):
        # initialize the stemmer and the lemamatizer
        self.Porter_Stemmer = PorterStemmer()
        self.Lancas_Stemmer = LancasterStemmer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
        
        # load the subtitile dictionary from the disk
        self.subtitledict = joblib.load(subdictpath)
        if (overwrite is True) or (not os.path.exists(worddictpath)):
            worddict = self.WordDict_Construct(self.subtitledict)
            self.worddict = worddict
            joblib.dump(self.worddict, worddictpath)
        elif os.path.exists(worddictpath):
            worddict = joblib.load(worddictpath)
            self.worddict = worddict
        else:
            print('please input the worddictpath or subdictpath')

    def WordDict_Construct(self, subtitledict):
        # initialize the worddict dict
        worddict = {}
        for key in subtitledict.keys():
            subdata = subtitledict[key]
            for index, data in enumerate(subdata):
                # data 的格式为 [beginindex, endindex, text]
                words = self.Split_Sentence2words(data[2], mode=3)
                for word in words:
                    if word in worddict.keys():
                        worddict[word].append([key, index])
                    else:
                        worddict[word] = [[key, index]]
        worddict = self.PurgeDict(worddict)
        return worddict
    
    # Split sentence to words
    def Split_Sentence2words(self, string, mode=3):
        string = string.lower()
        tokens = word_tokenize(string)
        if mode == 0:
            words = tokens
        elif mode == 1:
            words = [self.Porter_Stemmer.stem(word) for word in tokens]
        elif mode == 2:
            words = [self.Lancas_Stemmer.stem(word) for word in tokens]
        elif mode == 3:
            word_tags = [(word, self.get_wordnet_pos(tag)) for word, tag in pos_tag(tokens)]
            words = [self.wordnet_lemmatizer.lemmatize(word, pos=tag) for word, tag in word_tags]
        return words
    
    def PurgeDict(self, worddict, minNum=10):
        stop_words = stopwords.words('english')
        filter_words = []
        words = worddict.keys()
        for word in words:
            # remove the stopwords
            if word in stop_words:
                filter_words.append(word)
                continue
            # remove the rarely appear word
            if len(worddict[word]) < minNum:
                filter_words.append(word)
                continue
            # remove the others non-english word
            if word.find("'") != -1 or len(word) == 1:
                filter_words.append(word)
                continue
        for word in filter_words:
            worddict.pop(word)
        return worddict
    
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

    def ChooseSamples(self, word, delayfx, maxitems=500):
        begin_time = time.time()
        samples = []
        # 确定 positive samples 的 clip
        pos_instances = self.worddict[word]
        pos_instances = pos_instances[:min(len(pos_instances), maxitems)]
        
        for videokey, subindex in pos_instances:
            subtitledata = self.subtitledict[videokey]
            subinstance = subtitledata[subindex]
            # 将前一帧的起始位置作为候选的位置
            preindex = max(0, subindex-1)
            begin_index = subtitledata[preindex][0]

            # 确定结束的位置，保证要有足够的长度： 
            # addedlength > delayfx * currentlength
            bindex, eindex = subinstance[:2]
            framediff = eindex - bindex

            latterindex = subindex + 1
            while latterindex < len(subtitledata)-1:
                if subtitledata[latterindex][1] - eindex >= framediff * delayfx:
                    break
                latterindex += 1
            latterindex = min(latterindex, len(subtitledata)-1)
            end_index = subtitledata[latterindex][1]

            # 将数据加入到samples中
            samples.append([videokey, begin_index, end_index, 1])
        
        # 确定 negative samples 的样本
        # 将选用同样个数的 instance 作为最终的样本
        N = len(samples)
        numbers = 0

        # 确定 word 的 近义词集
        word_synsets = wordnet.synsets(word)
        synwords = set(chain.from_iterable([lemma.lemma_names() for lemma in word_synsets]))
        # counter = 0
        while True:
            # counter += 1
            videokey = np.random.choice(list(self.subtitledict.keys()))
            subtitledata = self.subtitledict[videokey]
            subindex = np.random.randint(len(subtitledata))
            # 一个negative sample 所划定的范围
            bindex = max(0, subindex-1)
            eindex = min(len(subtitledata)-1, subindex+1)

            Valid = True

            # 首先确定所选取的 subindex 和 positive samples 没有交集
            for i in range(N):
                if videokey == samples[i][0]:
                    if not (subtitledata[bindex][0] > samples[i][2] or subtitledata[eindex][1] < samples[i][1]):
                        Valid = False
                        break

            if Valid is False:
                # counter += 1
                continue

            # 然后保证该范围的字幕没有 Word 的近义词
            for i in range(bindex, eindex+1):
                # poter stemmer
                stem_words = self.Split_Sentence2words(subtitledata[i][2], mode=1)
                if self.Porter_Stemmer.stem(word) in stem_words:
                    Valid = False
                    break

                # lemmatizer
                lemma_words = self.Split_Sentence2words(subtitledata[i][2], mode=3)
                for s in lemma_words:
                    if s in synwords:
                        Valid = False
                        break
                
                if Valid is False:
                    break
            
            if Valid is False:
                continue
            samples.append([videokey, subtitledata[subindex][0], subtitledata[eindex][1], 0])
            numbers += 1
            if numbers == N:
                break

        print('the %s has %d samples, and consume %f seconds!' % (word, len(samples), time.time()-begin_time))
        # print('counter: %d, number: %d' % (counter, numbers))
        return samples

    def GetAvgFrames(self, subdata):
        # the subdata has length of M, where M is the subtitle instance numbers
        # each instance has the format as [begin, end, text]
        wordcounts = 0
        framecounts = 0
        for begin, end, text in subdata:
            framecounts += (end-begin)
            wordcounts += len(text.split())
        return framecounts/wordcounts
    
    def Infomation(self, *options):
        words = list(self.worddict.keys())
        words.sort(key=lambda x: len(self.worddict[x]), reverse=True)
        print('there are %d words in the worddictionary' % len(words))
        with open('../data/spbsl/wordinfo.txt', 'w') as f:
            for word in words:
                f.write('%s\t%d\n' % (word, len(self.worddict[word])))
        counters = [len(self.worddict[key]) for key in words]
        counters.sort()
        plt.plot(counters)
        plt.show()


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
    
    def Retrieve_Verification(self, word, indexes):
        if word not in self.annotation_dict.keys():
            print('the %s is not in the annotation dictionary')
            return 0

        # 获取真实的 word 的所在位置
        Records = self.annotation_dict[word]

        Verifications = np.zeros((len(Records),))
        for index, record in enumerate(Records):
            r_index, r_begin, r_end = record

            for i in range(len(indexes)):
                video_index, b_index, e_index = indexes[i]

                if r_index == video_index:
                    if r_begin > b_index and r_end < e_index:
                        Verifications[index] = 1
        return np.mean(Verifications)
     

def Test(testcode):
    # for the spbsl data
    subdictpath = '../data/spbsl/SubtitleDict.pkl'
    worddictpath = '../data/spbsl/WordDict.pkl'
    # annotationpath = '../data/bbc_sign_annotation.mat'
    
    # test the subtiledict class
    if testcode == 0:
        cl_worddict = WordsDict(worddictpath, subdictpath, overwrite=True)
        cl_worddict.ChooseSamples('predict', 1.5)
        # cl_worddict.Infomation()
    elif testcode == 1:
        pass


if __name__ == "__main__":
    Test(0)