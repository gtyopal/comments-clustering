# -*- coding: utf-8 -*-
import os
import sys
import dl_settings as config
from dl_train import train
import turicreate as tc
import pandas as pd
import time
from sklearn.cluster import KMeans,DBSCAN, MiniBatchKMeans
from sklearn import metrics
from sklearn.externals import joblib
from get_best_parameter import get_all_parameter
import hdbscan
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import numpy as np
from matplotlib import pyplot as plt


if not os.path.exists(config.TRAIN_DATA_all):
    print("Error! Feature does not exist, please run dl_train.py first")
else :
    tc_train_df = tc.SFrame.read_csv(config.TRAIN_DATA_all, header=False,skiprows=None)
    train_df = tc_train_df.to_dataframe()
    train_clean_df = pd.read_csv(config.TRAIN_DATA_CLEAN, encoding = "utf8")
    tc_clean = tc.SFrame(data=train_clean_df)
    all_paramter = get_all_parameter()
    t_start = time.time()

def make_center_tags(pd_data,cluster_ids = None,cluser_nums =None):
    resultdict = {}
    for index,center_ in enumerate(cluster_ids):
        real_cluster_id = cluster_ids[index]
        if cluser_nums[index] <= int(config.min_num_members_per_cluster):
            resultdict[str(real_cluster_id)] = -1
        else:
            resultdict[str(real_cluster_id)] = real_cluster_id
    pd_data['cluster_id'] = pd_data['cluster_id'].map(lambda x:resultdict[str(x)])
    return pd_data

def make_center_tags_others(pd_data):
    df_ids_counts = pd_data.groupby(['cluster_id']).count().iloc[:,0]
    ids = df_ids_counts.index.values
    resultdict = {"-1":-1}
    print(" make cluster tags files")
    for center_index,center_count in zip(ids,df_ids_counts):
        if center_count <= int(config.min_num_members_per_cluster):
            resultdict[str(center_index)] = -1
        else:
            resultdict[str(center_index)] = center_index
    pd_data['cluster_id'] = pd_data['cluster_id'].map(lambda x:resultdict[str(x)])
    return pd_data


if config.cluster_type == 'minibatchkmeans':
    feature_dim = train_df.shape[1]
    tmp = all_paramter["minibatchkmeans"]
    best_k = int(tmp)
    model = MiniBatchKMeans(n_clusters=int(best_k), init='k-means++', verbose=False, max_iter=200, n_init=30, batch_size=100)
    model.fit(train_df.values)
    joblib.dump(model, config.CLUSTER_MODEL)
    print("MinibatchKmeans clustering done!")
    predict_ids = model.labels_
    centroids = model.cluster_centers_
    print(len(centroids[0].tolist()))
    print(len(train_df.iloc[:, 1].tolist()))
    train_clean_df["cluster_id"] = predict_ids
    print("cluster number:" + str(len(set(predict_ids)) - (1 if -1 in predict_ids else 0)))
    cluster_df_sort = train_clean_df.sort_values(['cluster_id'], ascending=False)
    cluster_df_sort = make_center_tags_others(cluster_df_sort)
    print ("Generate output clustering matrix...")
    cluster_df_sort[["row_id", "message", "cluster_id"]].to_csv(config.cluster_tag_files, encoding="utf8", index=False)
    print("Time used: ", time.time() - t_start)
    si_score = metrics.silhouette_score(train_df, predict_ids)
    cal_score = metrics.calinski_harabaz_score(train_df, predict_ids)
    print('si_score=%2f, cal_score: %.2f' % (si_score, cal_score))


elif config.cluster_type == 'hdbscan':
    tmp = all_paramter["hdbscan"]
    min_samples = int(tmp)
    model = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=30, metric='euclidean', min_cluster_size= min_samples, min_samples= min_samples, p=None)
    model.fit(train_df.values)
    joblib.dump(model, config.CLUSTER_MODEL)
    print("hdbscan clustering done!")
    predict_ids = model.labels_
    train_clean_df["cluster_id"] = predict_ids
    print("cluster number:" + str(len(set(predict_ids)) - (1 if -1 in predict_ids else 0)))
    cluster_df_sort = train_clean_df.sort_values(['cluster_id'], ascending=False)
    print ("Generate output clustering matrix...")
    cluster_df_sort[["row_id","message","cluster_id"]].to_csv(config.cluster_tag_files, encoding = "utf8", index = False )
    print("Time used: ", time.time() - t_start)
    si_score = metrics.silhouette_score(train_df, predict_ids)
    cal_score = metrics.calinski_harabaz_score(train_df, predict_ids)
    print ('si_score=%2f, cal_score: %.2f' % (si_score, cal_score))


elif config.cluster_type == 'agglomerative':
    tmp = all_paramter["agglomerative"]
    best_k = int(tmp)
    current_model = estimator = AgglomerativeClustering(n_clusters=int(best_k), affinity='euclidean', linkage='ward')
    current_model.fit(train_df.values)
    joblib.dump(current_model, config.CLUSTER_MODEL)
    print("Hierachical clustering done!")
    predict_ids = current_model.labels_
    print("cluster number:" + str(len(set(predict_ids)) - (1 if -1 in predict_ids else 0)))
    train_clean_df["cluster_id"] = current_model.labels_
    cluster_df_sort = train_clean_df.sort_values(['cluster_id'],ascending=False)
    cluster_df_sort = make_center_tags_others(cluster_df_sort)
    print ("Generate output clustering matrix...")
    cluster_df_sort[["row_id", "message", "cluster_id"]].to_csv(config.cluster_tag_files, encoding="utf8", index=False)
    print("Time used: ", time.time() - t_start)
    si_score = metrics.silhouette_score(train_df, predict_ids)
    cal_score = metrics.calinski_harabaz_score(train_df, predict_ids)
    print ('si_score=%2f, cal_score: %.2f' % (si_score, cal_score))


data_2D = np.fromfile(config.TRAIN_DATA_all_2d, dtype=np.float32).reshape([-1, 2])
plt.scatter(data_2D[:, 0], data_2D[:, 1], c=predict_ids[:])
plt.title(config.cluster_type)
plt.savefig("dl_cluster.png")
plt.show()
