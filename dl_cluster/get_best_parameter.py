# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import dl_settings as config
import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans,DBSCAN, MiniBatchKMeans
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
from sklearn import metrics
from sklearn.externals import joblib
import hdbscan

def get_highest_score(score_dict):
    highest_key = 0
    highest_value = 0
    for key,value in score_dict.items():
        if value > highest_value:
            highest_value = value
            highest_key = key
    print(highest_key, highest_value)
    return highest_key

def get_model(modetype, model_paramter,df_features):
    if modetype == "minibatchkmeans":
        estimator = MiniBatchKMeans(n_clusters=int(model_paramter), init='k-means++', verbose=False, max_iter=200, n_init=30, batch_size=100)
    elif modetype == "hdbscan":
        estimator = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                    gen_min_span_tree=False, leaf_size=30, metric='euclidean', min_samples=int(model_paramter),min_cluster_size=int(model_paramter),  p=None)
    elif modetype == "agglomerative":
        ##C_matrix = kneighbors_graph(df_features.values, 2, mode='connectivity', include_self=True)
        ##estimator = AgglomerativeClustering(n_clusters=int(model_paramter), affinity='euclidean', linkage='ward',compute_full_tree=False,connectivity=C_matrix)
        estimator = AgglomerativeClustering(n_clusters=int(model_paramter), affinity='euclidean', linkage='ward')
    return estimator

# Get best n_cluster for Minibatchkmeans:
def get_minibatchkmeans_best_k(df_features, max_num_cluster):
    sse = []
    sse_d1 = []
    # Calinski-Harabaz Index: the larger the better
    calinski_score = {}
    Kcluster = list(range(5, max_num_cluster))
    for k in tqdm.tqdm(Kcluster):
        estimator = get_model('minibatchkmeans', k, df_features)  # build clustering estimator
        try:
            estimator.fit(df_features)
        except Exception:
            estimator.fit(df_features.values)
        sse.append(estimator.inertia_)
        calinski_score[str(k)] = metrics.calinski_harabaz_score(df_features, estimator.labels_)

    scalaer = MinMaxScaler(feature_range=(0, max_num_cluster))
    sse = scalaer.fit_transform(np.array(sse).reshape(-1, 1))
    sse_length = len(sse)
    for i in range(1, sse_length):
        sse_d1.append(sse[i][0] - sse[i - 1][0])
    sse_d1 = [i for i in sse_d1 if i < -1]
    bestK = Kcluster[sse_d1.index(max(sse_d1)) + 1]
    return bestK

# Get best eps and min_samples for hdbscan:
def get_hdbsan_best_eps_minsamples(df_features, max_num_members):

    calinski_score_dict = {}
    for i in range(2, max_num_members):
        min_samples = i
        min_cluster_size = min_samples
        estimator = get_model('hdbscan', min_samples, df_features)  # build clustering estimator
        estimator.fit(df_features)
        labels = estimator.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        try :
            calinski_score = metrics.calinski_harabaz_score(df_features, labels)
        except :
            calinski_score = 0
        print('min_cluster_size',min_cluster_size,'minsample', min_samples,'n_cluster: ',n_clusters_, 'calinski_score: ',calinski_score)
        calinski_score_dict[str(int(min_samples))] = calinski_score
    return get_highest_score(calinski_score_dict)

def get_Agglomerative_best_k(df_features,max_num_cluster):
    calinski_score_dict = {}
    Kcluster = list(range(5,max_num_cluster))
    for k in tqdm.tqdm(Kcluster):
        estimator = get_model('agglomerative', k, df_features)  # build clustering estimator
        labels = estimator.fit_predict(df_features)
        calinski_score = metrics.calinski_harabaz_score(df_features, labels)
        calinski_score_dict[k] = calinski_score
    return get_highest_score(calinski_score_dict)

def get_all_parameter():
    print("Start to search for best parameters")
    df_features = pd.read_csv(config.TRAIN_DATA_all, header=None)
    result_dict = {}

    if config.cluster_type == 'minibatchkmeans':
        print("Start getting minibatchkmeans parameter...")
        result_dict['minibatchkmeans'] = get_minibatchkmeans_best_k(df_features, max_num_cluster=config.max_num_cluster)
        print(result_dict['minibatchkmeans'])

    elif config.cluster_type == 'hdbscan':
        print("Start getting hdbscan parameter...")
        result_dict['hdbscan'] = get_hdbsan_best_eps_minsamples(df_features, max_num_members=int(config.max_num_cluster/2))
        print(result_dict['hdbscan'])

    elif config.cluster_type == 'agglomerative':
        print("Start getting agglomerative parameter...")
        result_dict['agglomerative'] = get_Agglomerative_best_k(df_features, max_num_cluster=config.max_num_cluster)
        print(result_dict['agglomerative'])

    return result_dict

def get_all_parameter_plot():
    print("Start to search for best parameters")
    df_features = pd.read_csv(config.TRAIN_DATA_all, header=None)
    result_dict = {}

    print("Start getting minibatchkmeans parameter...")
    result_dict['minibatchkmeans'] = get_minibatchkmeans_best_k(df_features, max_num_cluster=config.max_num_cluster)
    print(result_dict['minibatchkmeans'])


    print("Start getting hdbscan parameter...")
    result_dict['hdbscan'] = get_hdbsan_best_eps_minsamples(df_features, max_num_members=int(config.max_num_cluster/2))
    print(result_dict['hdbscan'])


    print("Start getting agglomerative parameter...")
    result_dict['agglomerative'] = get_Agglomerative_best_k(df_features, max_num_cluster=config.max_num_cluster)
    print(result_dict['agglomerative'])

    return result_dict