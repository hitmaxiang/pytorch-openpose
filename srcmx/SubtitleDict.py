'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-09-15 14:46:38
LastEditors: mario
LastEditTime: 2020-11-23 22:19:56
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


class WordsDict():
    def __init__(self, worddictpath, subdictpath=None, overwrite=False):
        if (overwrite is True) or (not os.path.exists(worddictpath)):
            subtitledict = joblib.load(subdictpath)
            worddict = self.WordDict_Construct(subtitledict)
            self.worddict = worddict
            self.save(worddictpath)
        elif os.path.exists(worddictpath):
            worddict = joblib.load(worddictpath)
            self.worddict = worddict
        else:
            print('please input the worddictpath or subdictpath')
    
    def WordDict_Construct(self, subtitledict):
        worddict = {}
        for key in subtitledict.keys():
            subdata = subtitledict[key]
            avgframe = self.GetAvgFrames(subdata)
            for index, data in enumerate(subdata):
                i = 1
                while index - i >= 0:
                    if data[0] - subdata[index-i][0] > 0.3 * avgframe * len(data[2].split()):
                        break
                    i += 1
                exbegin = subdata[max(index-i, 0)][0]
                j = 1
                while index + j < len(subdata):
                    if subdata[index+j][1] - data[1] > avgframe * len(data[2].split()):
                        break
                    j += 1
                exend = subdata[min(len(subdata)-1, index+j)][1]

                words = self.Split_Sentence2words(data[2], mode=3)
                for word in words:
                    if word in worddict.keys():
                        worddict[word].append([key, exbegin, exend, data[0], data[1], i, j])
                    else:
                        worddict[word] = [[key, exbegin, exend, data[0], data[1], i, j]]
        return worddict
    
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
        pattern = r"\b\w+[':]?\w*\b"
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
        else:
            # the default pos-tag is set to noun
            return wordnet.NOUN
    
    def save(self, outpath):
        joblib.dump(self.worddict, outpath)

    def ChooseSamples(self, word, extend=True):
        begin_time = time.time()
        samples = self.worddict[word]
        neg_samples = []
        neg_keys = list(self.worddict.keys())
        # np.random.seed(25)
        while len(neg_samples) < len(samples):
            neg_key = np.random.choice(neg_keys)
            if neg_key == word:
                continue
            rindex = np.random.randint(len(self.worddict[neg_key])) 
            neg_sample = self.worddict[neg_key][rindex]
            Valid = True

            # when the endfram is large than the framecount of the video, then it will be ignore
            videoindex, beginindex, endindex = neg_sample[:3]

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
    
    def GetAvgFrames(self, subdata):
        # the subdata has length of M, where M is the subtitle instance numbers
        # each instance has the format as [begin, end, text]
        wordcounts = 0
        framecounts = 0
        for begin, end, text in subdata:
            framecounts += (end-begin)
            wordcounts += len(text.split())
        return framecounts/wordcounts


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

    annotationpath = '../data/bbc_sign_annotation.mat'
    # test the subtiledict class
    if testcode == 0:
        cl_worddict = WordsDict(worddictpath, subdictpath, overwrite=False)
        print(cl_worddict.ChooseSamples('snow'))
    elif testcode == 1:
        anotationdict = AnnotationDict(annotationpath)

if __name__ == "__main__":
    Test(0)