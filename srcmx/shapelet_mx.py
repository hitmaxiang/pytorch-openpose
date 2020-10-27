'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-10-27 17:48:37
'''
import time
import utilmx
import joblib
import tslearn
import numpy as np
import tensorflow as tf
import PreprocessingData as PD
import matplotlib.pyplot as plt
from numba import jit
from tslearn import metrics
from tslearn.shapelets import LearningShapelets
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.preprocessing import scale
from SubtitleDict import SubtitleDict, AnnotationDict
from utilmx import Records_Read_Write


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
    
    def train(self, word=None, method=0):
        if word is None:
            words = []
            for keyword in self.cls_annotationdict.keys():
                if keyword in self.cls_subtitledict.keys():
                    words.append(keyword)
        else:
            words = [word]
        
        for word in words:
            self.word = word
            samples, pos_indexes = self.Getsamples(word)
            
            for m in range(4, 20):
                if method == 0:
                    self.FindShaplets_dtw_methods(samples, pos_indexes, m)
                elif method == 1:
                    self.FindShaplets_tslearn_class(samples, pos_indexes, m)
                elif method == 2:
                    self.FindShaplets_brute_force_ED(samples, pos_indexes, m)
    
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
        # tempscore, thre = self.Bipartition_score(best_locs[:, 0], len(pos_indexes), display=True)
        print('\n\n the length with % d and score is %f' % (m_len, tempscore))
        accuracy = self.RetriveAccuracy(pos_indexes, best_locs[:len(pos_indexes), 1:])
        print()
    
    def FindShaplets_tslearn_class(self, samples, pos_indexes, m_len, iters=300):
        '''
        description: using the shapelets class from tslearn to learn the shapelet
        param:
            samples: the list of samples, each sample with the shape with (n_times, n_dim)
        return: the shapelet
        author: mario
        '''
        # define the labels of the samples
        labels = np.zeros((len(samples),))
        labels[:len(pos_indexes)] = 1

        # prepare the samples to satisfy the format demand
        samples = tslearn.utils.to_time_series_dataset(samples)
        norm_samples = TimeSeriesScalerMinMax().fit_transform(samples)
        norm_samples = np.nan_to_num(norm_samples)

        # train and fit the shapelts_from_tslearn model
        shapelet_sizes = {m_len: 1}
        shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                    optimizer=tf.optimizers.Adam(.01),
                                    batch_size=16,
                                    weight_regularizer=0.01,
                                    max_iter=iters,
                                    random_state=42,
                                    verbose=0)
            
        shp_clf.fit(norm_samples, labels)

        # predict the samples
        score = shp_clf.score(norm_samples, labels)
        locations = shp_clf.locate(norm_samples[:len(pos_indexes)])
        
        Records_Read_Write().Write_shaplets_cls_Records(filepath='../data/record.txt', 
                                                        word=self.word,
                                                        m_len=m_len,
                                                        iters=iters,
                                                        featuremode=0,
                                                        score=score,
                                                        locs=locations)
        # self.RetriveAccuracy(pos_indexes, locations)
    
    def FindShaplets_brute_force_ED(self, samples, pos_indexes, m_len):
        
        begin_time = time.time()
        # 对样本集合进行归一化处理
        # samples = PD.NormlizeData(samples, mode=1)

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
                    # path, dis = metrics.dtw_subsequence_path(query, sample)
                    dis, loc = utilmx.Calculate_shapelet_distance(query, sample)
                    temp_record[sample_id] = np.array([dis, loc, loc+m_len])

                tempscore, thre = self.Bipartition_score(temp_record[:, 0], len(pos_indexes))
                if tempscore > best_score:
                    best_score = tempscore
                    best_query = query
                    best_locs = temp_record
        
        print('each sample cost time %f seconds' % (time.time()-begin_time))
        # tempscore, thre = self.Bipartition_score(best_locs[:, 0], len(pos_indexes), display=True)
        print('\n\n the length with % d and score is %f' % (m_len, tempscore))
        accuracy = self.RetriveAccuracy(pos_indexes, best_locs[:len(pos_indexes), 1:])
        print(accuracy)

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
    
    def RetriveAccuracy(self, sample_indexes, locations):
        distances = self.cls_annotationdict.Retrive_distance(self.word, sample_indexes, locations)

        correct_num = 0
        for dis in distances:
            if dis == 0:
                correct_num += 1
        accueacy = correct_num/len(distances)

        print(distances)
        print(accueacy)
        
        return accueacy
    
    def RetriveAccuracy_with_record_file(self, word, filepath):
        self.word = word
        best_accuracy = 0
        best_arg = []
        pos_indexes = self.Getsamples(word)[1]
        record_dict = Records_Read_Write().Read_shaplets_cls_Records(filepath)
        temp_record = np.zeros((len(pos_indexes), 2))
        if self.word in record_dict.keys():
            key_record_dict = record_dict[self.word]
            for key_args in key_record_dict.keys():
                m_len = int(key_args.split('-')[0])
                for record in key_record_dict[key_args]['location']:
                    temp_record[:, 0] = np.array(record)
                    temp_record[:, 1] = np.array(record) + m_len
                    accuracy = self.RetriveAccuracy(pos_indexes, temp_record)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_arg = [key_args]
                    elif accuracy == best_accuracy:
                        best_arg.append(key_args)
        print(best_accuracy, best_arg)
            
    def Retrieve_Verification(self):
        words = self.cls_annotationdict.annotation_dict.keys()
        for word in words:
            if word not in self.cls_subtitledict.subtitledict.keys():
                continue
            pos_indexes, neg_indexes = self.cls_subtitledict.ChooseSamples(word)
            indexes = np.array(pos_indexes)[:, :3]
            # verification test the real word is in the candidate clips
            retrieve_accuracy = self.cls_annotationdict.Retrieve_Verification(word, indexes)
            print('the verification of %s is %f\n' % (word, retrieve_accuracy))

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
        cls_shapelet.train('work', method=1)
        # cls_shapelet.RetriveAccuracy_with_record_file('snow', '../data/record_server.txt')
        # cls_shapelet.Retrieve_Verification()
        

if __name__ == "__main__":
    Test(1)