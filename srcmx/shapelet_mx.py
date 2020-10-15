'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-10-15 22:13:48
'''
import time
import utilmx
import joblib
import numpy as np
import PreprocessingData as PD
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from tslearn import metrics
from numba import jit
from SubtitleDict import SubtitleDict, AnnotationDict


class Shapelets_mx():
    def __init__(self, motion_dictpath, subtitle_dictpath, annotation_dictpath):
        self.motiondatadict = joblib.load(motion_dictpath)
        self.cls_subtitledict = SubtitleDict(subtitle_dictpath)
        self.cls_annotationdict = AnnotationDict(annotation_dictpath)
    
    def Getsamples(self, word):
        '''
        description: get the instance of the word, and random sample the negative samples
        param: word, the queried word
        return: pos_indexes, neg_indexes, pos_samples, neg_samples
        author: mario
        '''
        # 抽样得到 pos 以及 neg 的样本的索引以及clip位置
        pos_indexes, neg_indexes = self.cls_subtitledict.ChooseSamples(word)
        samples = []
        clip_indexes = np.concatenate((pos_indexes, neg_indexes), axis=0)

        # 从 motiondict 中 按照上面得到的索引位置提取数据
        for record in clip_indexes:
            videoindex, beginindex, endindex = record[:3]
            videokey = '%dvideo' % (videoindex+1)
            if videokey not in self.motiondatadict.keys():
                continue
            clip_data = self.motiondatadict[videokey][0][beginindex:endindex]

            # 针对每个 clip 数据, 只选取上面身的关节数据作为特征
            clip_data = PD.MotionJointSelect(clip_data, datamode='body', featuremode=0)
            clip_data = np.reshape(clip_data, (clip_data.shape[0], -1))
            samples.append(clip_data)
            
        return samples, clip_indexes[:len(pos_indexes), :2]
    
    def train(self, word):
        self.word = word
        samples, pos_indexes = self.Getsamples(word)
        self.FindShaplets_dtw_methods(samples, pos_indexes, 10)
    
    def FindShaplets_dtw_methods(self, samples, pos_indexes, m_len):
        # 对样本集合进行归一化处理
        samples = PD.NormlizeData(samples, mode=0)

        # 设置最后保留的 shapelet
        best_score = 0
        best_query = None
        best_locs = None

        # 从所有的 pos_sample 中提取所有可能的 query 子序列
        for sample_id in range(len(pos_indexes)):
            pos_sample = samples[sample_id]
            for q_index in range(len(pos_sample)-m_len+1):
                query = pos_sample[q_index:q_index+m_len]

                # 使用该 query 对所有样本数据进行距离求取
                temp_record = np.zeros((len(samples), 3))
                for sample_id in range(len(samples)):
                    sample = samples[sample_id]
                    path, dis = metrics.dtw_subsequence_path(query, sample)
                    temp_record[sample_id] = np.array([dis, path[0][1], path[-1][1]])

                tempscore, thre = self.Bipartition_score(temp_record[:, 0], len(pos_indexes))
                if tempscore > best_score:
                    best_score = tempscore
                    best_query = query
                    best_locs = temp_record
        tempscore, thre = self.Bipartition_score(best_locs[:, 0], len(pos_indexes), display=True)
        print()
    
    def Bipartition_score(self, distances, pos_num, display=False):
        '''
        description: 针对一个 distances 的 二分类的最大分类精度
        param: 
            pos_num: 其中 distance 的前 pos_num 个 的标签为 1, 其余都为 0
        return: 最高的分类精度, 以及对应的分割位置
        author: mario
        '''     
        dis_sort_index = np.argsort(distances)
        correct = len(distances) - pos_num
        Bound_correct = len(distances)
        maxcorrect = correct
        maxindex = 0
        for i, index in enumerate(dis_sort_index):
            if index < pos_num:  # 分对的
                correct += 1
                if correct > maxcorrect:
                    maxcorrect = correct
                    maxindex = i+1
            else:
                correct -= 1
                Bound_correct -= 1
            if correct == Bound_correct:
                break
            
        if maxindex != 0:
            thre = (distances[dis_sort_index[maxindex-1]] + distances[dis_sort_index[maxindex]])/2
        else:
            thre = distances[dis_sort_index[maxindex]]
        
        score = maxcorrect/len(distances)

        if display:
            plt.scatter(np.arange(0, pos_num), distances[:pos_num])
            plt.scatter(np.arange(pos_num, len(distances)), distances[pos_num:])
            plt.plot([0, len(distances)], [thre, thre])
            plt.title('%f' % score)
            plt.show()
        
        return score, thre


def Shapelets_Rangelength(pos_indexes, pos_samples, neg_samples, lengthrange):
    '''
    description: 
    param {type} 
    return {type} 
    author: mario
    ''' 
    low_len, high_len = lengthrange
    labels = [1] * (len(pos_samples)-1) + [0] * len(neg_samples)
    for length in range(low_len, high_len+1):
        maxrecord = [0, 0, 0, 0]  # length, index, qindex, score
        begintime = time.time()
        for index, pos_sample in enumerate(pos_samples):
            
            testsamples = pos_samples[:index] + pos_samples[index+1:] + neg_samples
            for q_index in range(len(pos_sample)-length+1):
                query = pos_sample[q_index:q_index+length]
                score = ShapeletsScore(query, testsamples, labels)
                if score > maxrecord[-1]:
                    maxrecord = [length, index, q_index, score]

        print(maxrecord, 'with time %f' % (time.time()-begintime))


# @jit
def ShapeletsScore(query, samples, labels):
    Distances = np.zeros((len(samples),))
    Locations = []  # np.zeros((len(samples), len(query), 2))
    for index, sample in enumerate(samples):
        locas, score = metrics.dtw_subsequence_path(query, sample)
        Distances[index] = score
        Locations.append(locas)

    # caculate the optimal classfication rate
    dis_sort_index = np.argsort(Distances)
    correct = len(labels) - sum(labels)
    Bound_correct = len(labels)
    maxcorrect = correct
    for idnum in dis_sort_index:
        if labels[idnum] == 1:  # 分对的
            correct += 1
            if correct > maxcorrect:
                maxcorrect = correct
        else:
            correct -= 1
            Bound_correct -= 1
        if correct == Bound_correct:
            break
        # print('%d-%d-%f' % (Bound_correct, correct, correct/len(labels)))
    return(maxcorrect/len(labels))
    

def Test_Whole_Route(word, motiondict, cls_subtitledict, cls_annotationdict):
    
    # prepare the test data
    pos_indexes, neg_indexes = cls_subtitledict.ChooseSamples(word)
    Lengths = []
    samples = []
    clip_indexes = pos_indexes + neg_indexes
    for record in clip_indexes:
        videoindex, beginindex, endindex = record[:3]
        # the begin index of the dict is 1
        videokey = '%dvideo' % (videoindex+1)
        if videokey not in motiondict.keys():
            continue
        clip_data = motiondict[videokey][0][beginindex:endindex]

        Lengths.append(clip_data.shape[0])
        # select the defined motion joint
        clip_data = PD.MotionJointSelect(clip_data, datamode='body', featuremode=0)
        clip_data = np.reshape(clip_data, (clip_data.shape[0], -1))
        # normlize the clipdata
        clip_data = scale(clip_data)
        samples.append(clip_data)
    

def Test(testcode):
    subtitledictpath = '../data/subtitledict.pkl'
    motionsdictpath = '../data/motionsdic.pkl'
    annotationdictpath = '../data/annotationdict.pkl'

    if testcode == 0:
        pos_samples = [np.random.randint(10, size=(np.random.randint(800, 1000), 4)) for i in range(100)]
        neg_samples = [np.random.randint(10, size=(np.random.randint(800, 1000), 4)) for i in range(100)]
        Shapelets_Rangelength(pos_samples, neg_samples, (10, 20))

    elif testcode == 1:
        cls_shapelet = Shapelets_mx(motionsdictpath, subtitledictpath, annotationdictpath)
        cls_shapelet.train('snow')
        


if __name__ == "__main__":
    Test(1)