'''
Description: 
Version: 2.0
Autor: mario
Date: 2020-09-22 20:45:10
LastEditors: mario
LastEditTime: 2020-09-24 01:44:03
'''
import tslearn
import numpy as np
import tensorflow as tf
import MotionEstimation as ME
import matplotlib.pyplot as plt
from SubtitleDict import SubtitleDict
from tslearn.utils import ts_size
from sklearn.metrics import accuracy_score
from tslearn.utils import to_time_series_dataset
from tslearn.shapelets import LearningShapelets, grabocka_params_to_shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMinMax


def GetShapelets(motiondict, subtitledic, word, display=False):
    pos_indexes, neg_indexes = subtitledic.ChooseSamples(word)
    samples = []
    labels = [1] * len(pos_indexes) + [0] * len(neg_indexes)
    clip_indexes = pos_indexes + neg_indexes
    Lengths = []
    
    for record in clip_indexes:
        videoindex, beginindex, endindex = record[:3]
        videoindex += 1  # the begin index of the dict is 1
        if videoindex not in motiondict.keys():
            continue
        clip_data = motiondict[videoindex][0][beginindex:endindex]
        
        Lengths.append((endindex-beginindex, clip_data.shape[0]))
        if display:
            print('the clip is the frame: %d--%d of file:e-%d.avi, the length is:' % 
                  (beginindex, endindex, videoindex), Lengths[-1])
            ME.Demons_SL_video_clip(record, clip_data)
        samples.append(clip_data)
    # in this function, the data should be padding
    maxlen = max([x[0] for x in Lengths])
    for index, clip_data in enumerate(samples):
        clip_data = np.pad(clip_data, ((0, maxlen-clip_data.shape[0]), (0, 0), (0, 0)))
        clip_data = np.reshape(clip_data, (clip_data.shape[0], 18*2))
        samples[index] = clip_data
    # score_data = motiondict[videoindex][1][beginindex:endindex]
        
    # filling NAN to the samples to have the same times length
    samples = np.array(samples)
    # before the shapelest learn, normalize the sameples should be done
    # samples = TimeSeriesScalerMinMax().fit_transform(samples)

    n_ts, ts_sz = samples.shape[:2]
    shapelet_sizes = grabocka_params_to_shapelet_size_dict(n_ts=n_ts, ts_sz=ts_sz, n_classes=2, l=0.025, r=1)
    shp_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                optimizer=tf.optimizers.Adam(.01),
                                batch_size=16,
                                weight_regularizer=0.01,
                                max_iter=200,
                                random_state=42,
                                verbose=0)
    shp_clf.fit(samples, labels)

    # Plot the different discovered shapelets
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" % (shapelet_sizes[sz], sz))
        for shp in shp_clf.shapelets_:
            if ts_size(shp) == sz:
                plt.plot(shp.ravel())
        plt.xlim([0, max(shapelet_sizes.keys()) - 1])

    plt.tight_layout()
    # plt.show()

    # The loss history is accessible via the `model_` that is a keras model
    plt.figure()
    plt.plot(np.arange(1, shp_clf.n_iter_ + 1), shp_clf.history_["loss"])
    plt.title("Evolution of cross-entropy loss during training")
    plt.xlabel("Epochs")
    plt.show()


def Test(code):
    import joblib
    subtitledictpath = '../data/subtitledict.pkl'
    motionsdictpath = '../data/motionsdic.pkl'
    code = 0
    if code == 0:
        subtitledict = SubtitleDict(subtitledictpath)
        motionsdict = joblib.load(motionsdictpath)
        GetShapelets(motionsdict, subtitledict, 'snow')


if __name__ == "__main__":
    Test(0)
