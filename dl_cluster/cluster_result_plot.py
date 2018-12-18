# coding: utf-8
import os
import sys
PROJECT_ROOT =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from sklearn import metrics
from sklearn.cluster import KMeans,DBSCAN, MiniBatchKMeans, AgglomerativeClustering
import pandas as pd
import dl_settings as config
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt
from dl_cluster.get_best_parameter import get_all_parameter_plot
import os
import hdbscan

print(" use all data to cluster!")
if (not os.path.exists(config.TRAIN_DATA_all)) :
    raise ValueError("please rnn data_preprocessing first")
train_data_df = pd.read_csv(config.TRAIN_DATA_all,header=None)
print('Total message numbers: ', train_data_df.shape[0])
data_2D = np.fromfile(config.TRAIN_DATA_all_2d, dtype=np.float32).reshape([-1, 2])

def minibatchkmeans_score_metrics(best_k):
    model = MiniBatchKMeans(n_clusters=int(best_k), init='k-means++', verbose=False, max_iter=200, n_init=30, batch_size=100)
    model.fit(train_data_df)
    predict_ids = model.fit_predict(train_data_df)
    si_score = metrics.silhouette_score(train_data_df, predict_ids)
    cal_score = metrics.calinski_harabaz_score(train_data_df, predict_ids)
    return  predict_ids,si_score,cal_score

def hdbscan_score_metrics(min_samples):
    model = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=30, metric='euclidean', min_cluster_size= min_samples, min_samples= min_samples, p=None)
    model.fit(train_data_df)
    predict_ids = model.labels_
    si_score = metrics.silhouette_score(train_data_df, predict_ids,metric='cosine')
    cal_score = metrics.calinski_harabaz_score(train_data_df, predict_ids)
    return predict_ids, si_score, cal_score

def agglomerative_score_metrics(best_k):
    model = AgglomerativeClustering(n_clusters=best_k, affinity='euclidean', linkage='ward')
    ###model = AgglomerativeClustering(n_clusters=int(model_paramter), affinity='cosine', linkage='average')
    predict_ids = model.fit_predict(train_data_df)
    si_score = metrics.silhouette_score(train_data_df, predict_ids)
    cal_score = metrics.calinski_harabaz_score(train_data_df, predict_ids)
    return  predict_ids,si_score,cal_score

def get_score_metrics():
    all_paramter = get_all_parameter_plot()
    print("best paramter:")
    pp.pprint(all_paramter)
    best_k = int(all_paramter['minibatchkmeans'])
    plt.subplot(221)
    predict_ids, si_score, cal_score = minibatchkmeans_score_metrics(best_k)
    plt.scatter(data_2D[:,0], data_2D[:,1], c=predict_ids.tolist()[:])
    plt.text(.99, .01, ('si_score=%2f, cal_score: %.2f' % (si_score, cal_score)),transform=plt.gca().transAxes, size=10,horizontalalignment='right')
    plt.title('minibatchkmeans')

    min_samples = int(all_paramter['hdbscan'])
    plt.subplot(222)
    predict_ids, si_score, cal_score = hdbscan_score_metrics(min_samples)
    plt.scatter(data_2D[:, 0], data_2D[:, 1], c=predict_ids[:])
    plt.text(.99, .01, ('si_score=%2f, cal_score: %.2f' % (si_score, cal_score)),transform=plt.gca().transAxes, size=10,horizontalalignment='right')
    plt.title('hdbscan')

    plt.subplot(223)
    predict_ids, si_score, cal_score = minibatchkmeans_score_metrics(int(all_paramter['agglomerative']))
    plt.scatter(data_2D[:,0], data_2D[:,1], c=predict_ids.tolist()[:])
    plt.text(.99, .01, ('si_score=%2f, cal_score: %.2f' % (si_score, cal_score)),transform=plt.gca().transAxes, size=10,horizontalalignment='right')
    plt.title('agglomerative')

    plt.tight_layout()
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    get_score_metrics()


