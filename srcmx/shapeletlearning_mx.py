'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: mario
LastEditTime: 2020-12-07 20:45:15
'''
import time
import utilmx
import joblib
import tslearn
import torch
import numpy as np
import PreprocessingData as PD
import matplotlib.pyplot as plt
from numba import jit
from SubtitleDict import WordsDict, AnnotationDict
from utilmx import Records_Read_Write


class Shapelets_mx():
    def __init__(self, motion_dictpath, word_dictpath, subtitle_dictpath, annotation_dictpath=None):
        self.motiondatadict = joblib.load(motion_dictpath)
        self.cls_worddict = WordsDict(word_dictpath, subtitle_dictpath)
        if annotation_dictpath is not None:
            self.cls_annotationdict = AnnotationDict(annotation_dictpath)
        else:
            self.cls_annotationdict = None
    
    def Getsamples(self, word):
        '''
        description: get the instance of the word, and random sample the negative samples
        param: word, the queried word
        return: pos_indexes, neg_indexes, pos_samples, neg_samples
        author: mario
        '''
        # 抽样得到 pos 以及 neg 的样本的索引以及clip位置
        # sample_indexes 的格式为：[videokey(str), begin, end, label]
        sample_indexes = self.cls_worddict.ChooseSamples(word, 1.5)
        samples = []

        # 从 motiondict 中 按照上面得到的索引位置提取数据
        # motiondata format: [motiondatas, scores]
        for i in range(len(sample_indexes)):
            videokey, beginindex, endindex = sample_indexes[i][:3]
            if videokey not in self.motiondatadict.keys():
                continue
            clip_data = self.motiondatadict[videokey][0][beginindex:endindex]

            # 针对每个 clip 数据, 只选取上面身的关节数据作为特征
            clip_data = PD.MotionJointSelect(clip_data, datamode='body', featuremode=0)
            clip_data = np.reshape(clip_data, (clip_data.shape[0], -1))
            # 因为原始的数据类型为int16， 在后续计算的过程中，容易溢出
            clip_data = clip_data.astype(np.float32)
            samples.append(clip_data)
        return samples, sample_indexes
    
    def train(self, word=None, method=2):
        if word is None:
            words = self.cls_worddict.worddict.keys()
        elif isinstance(word, str):
            words = [word]
        elif isinstance(word, list):
            words = word
        
        trainedrecords = utilmx.ReadShapeletRecords('../data/spbsl/shapeletED.txt')

        for word in words:
            # 现阶段，对于sample特别多的先不分析
            if len(self.cls_worddict.worddict[word]) >= 500:
                continue
            if word in trainedrecords.keys():
                if len(trainedrecords[word]) == 26:
                    continue
            self.word = word
            samples, sample_indexes = self.Getsamples(word)
            
            for m in range(4, 30):
                if word in trainedrecords.keys():
                    if m in trainedrecords[word]:
                        continue
                if method == 0:
                    # self.FindShaplets_dtw_methods(samples, sample_indexes, m)
                    pass
                elif method == 1:
                    # self.FindShaplets_tslearn_class(samples, pos_indexes, m)
                    pass
                elif method == 2:
                    # self.FindShaplets_brute_force_ED(samples, sample_indexes, m)
                    self.FindShaplets_brute_force_ED_torch(samples, sample_indexes, m)
    
    def FindShaplets_dtw_methods(self, samples, pos_indexes, m_len):
        pass
    
    def FindShaplets_brute_force_ED(self, samples, sample_indexes, m_len):
        
        begin_time = time.time()
        # 对样本集合进行归一化处理
        # samples = PD.NormlizeData(samples, mode=1)
        labels = [x[-1] for x in sample_indexes]

        BestKshapelets = utilmx.BestKItems(K=10)
        # 设置最后保留的 shapelet
        N = len(samples)
        # 使用两级优化的方式进行优化
        for i in range(N):
            if sample_indexes[i][-1] == 0:
                continue
            l_1 = len(samples[i])
            Dis_sample = np.zeros((N, l_1-m_len+1))
            Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
            
            for j in range(N):
                if i == j:
                    Dis_sample[i] = 0
                    Dis_loc[i] = np.arange(Dis_loc.shape[1])
                    continue
                DisMat = utilmx.matrixprofile(samples[i], samples[j], m_len)
                Dis_loc[j] = np.argmin(DisMat, axis=-1)
                Dis_sample[j] = np.min(DisMat, axis=-1)
            
            # 针对每一个可能的 candidate sign, 求解它的score
            for candin_index in range(l_1-m_len+1):
                score, thre = self.Bipartition_score(Dis_sample[:, candin_index], sum(labels))
                key = '%s-framindex:%d-offset:%d-m_len:%d' % (sample_indexes[i][0], sample_indexes[i][1], candin_index, m_len)
                data = Dis_loc[:, candin_index]
                BestKshapelets.insert(score, [key, data])
        
        Headerinfo = 'the word:%s with m length: %d' % (self.word, m_len)
        BestKshapelets.wirteinfo(Headerinfo, '../data/spbsl/shapeletED.txt', 'a')
        print('%d samples with %d m_len cost time %f seconds' % (N, m_len, (time.time()-begin_time)))

    def FindShaplets_brute_force_ED_torch(self, samples, sample_indexes, m_len):
        
        begin_time = time.time()
        # 对样本集合进行归一化处理
        # samples = PD.NormlizeData(samples, mode=1)
        N = len(samples)
        BestKshapelets = utilmx.BestKItems(K=10)

        lengths = [0] + [len(x) for x in samples] 
        cumlength = np.cumsum(lengths)

        labels = [x[-1] for x in sample_indexes]
        samples = [torch.from_numpy(x) for x in samples]
        catsamples = torch.cat(samples, dim=0)
        if torch.cuda.is_available():
            catsamples = catsamples.cuda()

        # 不可以用全体的方式进行，需要使用分批次的方式进行
        # with torch.no_grad():
        #     DISMAT = utilmx.matrixprofile_torch(catsamples[:cumlength[sum(labels)]], catsamples, m_len)
        #     # DISMAT = utilmx.matrixprofile_torch(catsamples, catsamples, m_len)
        # DISMAT = DISMAT.cpu().numpy()
        # torch.cuda.empty_cache()

        for i in range(sum(labels)):
            l_1 = lengths[i+1]
            Dis_sample = np.zeros((len(samples), l_1-m_len+1))
            Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
            
            begin = cumlength[i]
            end = begin + lengths[i+1]
            with torch.no_grad():
                DISMAT = utilmx.matrixprofile_torch(catsamples[begin:end], catsamples, m_len)
            DISMAT = DISMAT.cpu().numpy()

            for j in range(len(samples)):
                index_by = cumlength[j]
                index_ey = index_by + lengths[j+1] - m_len + 1

                DisMat = DISMAT[:, index_by:index_ey]
                Dis_loc[j] = np.argmin(DisMat, axis=-1)
                Dis_sample[j] = np.min(DisMat, axis=-1)

            # 针对每一个可能的 candidate sign, 求解它的score
            for candin_index in range(l_1-m_len+1):
                score, thre = self.Bipartition_score(Dis_sample[:, candin_index], sum(labels))
                key = '%s-framindex:%d-offset:%d-m_len:%d' % (sample_indexes[i][0], sample_indexes[i][1], candin_index, m_len)
                data = Dis_loc[:sum(labels), candin_index]
                BestKshapelets.insert(score, [key, data])
        
        Headerinfo = 'the word:%s with m length: %d' % (self.word, m_len)
        BestKshapelets.wirteinfo(Headerinfo, '../data/spbsl/shapeletED.txt', 'a')
        print('%d samples with %d m_len cost time %f seconds' % (N, m_len, (time.time()-begin_time)))

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


def Test(testcode):
    subtitledictpath = '../data/subtitledict.pkl'
    motionsdictpath = '../data/motionsdic.pkl'
    # annotationdictpath = '../data/annotationdict.pkl'

    if testcode == 0:
        motionsdictpath = '../data/spbsl/motionsdic.pkl'
        worddictpath = '../data/spbsl/WordDict.pkl'
        subtitledictpath = '../data//spbsl/SubtitleDict.pkl'
        cls_shapelet = Shapelets_mx(motionsdictpath, worddictpath, subtitledictpath)
        # consider: 500, thank:2153
        # cls_shapelet.train('thank', method=2)
        cls_shapelet.train(method=2)

        
if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    Test(2)