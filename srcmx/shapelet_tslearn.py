'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-09-22 20:45:10
LastEditors: mario
LastEditTime: 2020-10-13 13:14:27
'''
import tslearn
import time
import utilmx
import multiprocessing
import numpy as np
import tensorflow as tf
import MotionEstimation as ME
import PreprocessingData as PD
import matplotlib.pyplot as plt
from SubtitleDict import SubtitleDict
from tslearn.utils import ts_size
from sklearn.metrics import accuracy_score
from tslearn.utils import to_time_series_dataset
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMinMax


def GetShapelets(motiondict, pos_indexes, neg_indexes, word, iters, s_length, featuremode, display=False):
    # pos_indexes, neg_indexes = subtitledic.ChooseSamples(word)
    samples = []
    labels = [1] * len(pos_indexes) + [0] * len(neg_indexes)
    clip_indexes = pos_indexes + neg_indexes
    Lengths = []
    for record in clip_indexes:
        videoindex, beginindex, endindex = record[:3]
        # the begin index of the dict is 1
        videokey = '%dvideo' % (videoindex+1)
        if videokey not in motiondict.keys():
            continue
        clip_data = motiondict[videokey][0][beginindex:endindex]

        # demostrate the videoclip and poseclip
        if display:
            print('the clip is the frame: %d--%d of file:e-%d.avi, the length is%d--%d:' % 
                  (beginindex, endindex, videoindex+1, endindex - beginindex, Lengths[-1]))
            ME.Demons_SL_video_clip(record, clip_data)

        # judge whether the clip length is correct
        Lengths.append(clip_data.shape[0])
        if clip_data.shape[0] != (endindex-beginindex):
            print('the length of the motion clip and subtitle clip is not equal!')
        
        # select the defined motion joint
        clip_data = PD.MotionJointSelect(clip_data, datamode='body', featuremode=featuremode)
        samples.append(clip_data)
    # the data should be padding to the maxlength
    maxlen = max(Lengths)
    for index, clip_data in enumerate(samples):
        clip_data = np.pad(clip_data, ((0, maxlen-clip_data.shape[0]), (0, 0), (0, 0)))
        samples[index] = np.reshape(clip_data, (clip_data.shape[0], -1))
    samples = np.array(samples)
    
    # before the shapelest learn, normalize the sameples should be done
    norm_samples = TimeSeriesScalerMinMax().fit_transform(samples)
    norm_samples = np.nan_to_num(norm_samples)
    n_ts, ts_sz = norm_samples.shape[:2]
    shapelet_sizes = {s_length: 1}
    # shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=2, l=0.025, r=1)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=0.01,
                                max_iter=iters,
                                random_state=42,
                                verbose=0)
    
    # begin_time = time.time()
    shp_clf.fit(norm_samples, labels)
    # print('the model train consume %f seconds' % (time.time()-begin_time))
    
    # shp_clf.to_pickle('../data/%s.pkl' % word)
    score = shp_clf.score(norm_samples, labels)
    Locs = shp_clf.locate(norm_samples[: len(pos_indexes)])
    # begin_time = time.time()
    with open('../data/record.txt', 'a+') as f:
        f.write('\nthe test of word: %s, with lenth: %d, iters: %d, feature: %d\n' %
                (word, s_length, iters, featuremode))
        f.write('score:\t%f\n' % score)
        f.write('Locs:')
        for loc in Locs:
            f.write('\t%d' % loc[0])
        f.write('\n\n')
    # print('the file write consume %f seconds' % (time.time()-begin_time))
    print('the score is %f' % shp_clf.score(norm_samples, labels))

    # for key, lossdata in shp_clf.history_.items():
    #     plt.figure()
    #     plt.plot(np.arange(1, shp_clf.n_iter_ + 1), lossdata)
    #     plt.title("Evolution of %s during training" % key)
    #     plt.xlabel("Epochs")
    # plt.show()


def Test(code):
    import joblib
    subtitledictpath = '../data/subtitledict.pkl'
    motionsdictpath = '../data/motionsdic.pkl'
    code = 0
    if code == 0:
        # P = multiprocessing.Pool(processes=4)
        subtitledict = SubtitleDict(subtitledictpath)
        motionsdict = joblib.load(motionsdictpath)
        recorddict = utilmx.ReadRecord('../data/record.txt')

        Batch_size = 50 
        word = 'snow'
        ITERS = [300, 400, 500, 600, 700, 800, 900]
        S_LENGTH = [i for i in range(4, 20)]
        for lens in S_LENGTH:
            for iters in ITERS:
                for feature in [0, 1]:
                    key = '%d-%d-%d' % (lens, iters, feature)
                    print(key)
                    deficit_repeat = 50
                    if key in recorddict.keys():
                        deficit_repeat -= recorddict[key]['num']
                    if deficit_repeat > 0: 
                        for i in range(deficit_repeat):
                            pos_indexes, neg_indexes = subtitledict.ChooseSamples(word)
                            # P.apply_async(GetShapelets, args=(motionsdict, subtitledict, 'snow', iters, lens, feature))
                            GetShapelets(motionsdict, pos_indexes, neg_indexes, word, iters, lens, feature)
                            Batch_size -= 1
                            print('the iterations left: %d' % Batch_size)
                            if Batch_size <= 0:
                                break
                    if Batch_size <= 0:
                        break
                if Batch_size <= 0:
                    break
            if Batch_size <= 0:
                break
        # P.close()
        # P.join()


if __name__ == "__main__":
    Test(0)
