# coding: utf-8
import os
import sys
PROJECT_ROOT =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import config.dl_settings as config
import dl_cluster.dl_data_utils as data_utils
import tensorflow as tf
import dl_cluster.dl_model as modelSeq
import numpy as np
import turicreate as tc
from sklearn.externals import joblib
from dl_cluster.get_best_parameter import get_model
import dl_cluster.data_preprocessing
from dl_cluster.data_preprocessing import get_vec_with_models, get_statis
import pandas as pd

def predict_clusters_id(text):
    test_word2vec = get_vec_with_models( [text], "word2vec")
    test_lsi = get_vec_with_models([text], "lsi")
    test_seq2seq = get_vec_with_models([text], "seq2seq")
    df_w2v_feature = pd.DataFrame(test_word2vec)
    df_lsi_feature = pd.DataFrame( test_lsi)
    df_seq2seq_feature = pd.DataFrame( test_seq2seq)
    word2vec_matrix = get_statis(df_w2v_feature)
    lsi_matrix = get_statis(df_lsi_feature)
    seq2seq_matrix = get_statis(df_seq2seq_feature)
    test_vec = ""
    if config.model_has_used == "seq2seq_lsi_word2vec":
        test_vec = np.hstack((word2vec_matrix, lsi_matrix,seq2seq_matrix))
    if config.model_has_used == "seq2seq_lsi":
        test_vec = np.hstack((lsi_matrix,seq2seq_matrix))
    if config.model_has_used == "seq2seq":
        test_vec = np.array(seq2seq_matrix)
    ### test_vec = np.hstack((word2vec_matrix,lsi_matrix,seq2seq_matrix))
    test_vec_pca = ""
    if (not os.path.exists(config.scalar_model_path) ) :
        print("model is not exist,please set the config.currentMode to train  and run data_processing.py first")
        exit()
    else:
        scalar = joblib.load( config.scalar_model_path ) 
        test_vec_scala = scalar.transform(test_vec)
        test_vec_pca = test_vec_scala
        if config.used_pca == True:
            if not os.path.exists(config.pca_model_path) :
                print("pca model is not exist,please set the config.used_pca to True  and run data_preprocessing first")
                exit()
            pca = joblib.load( config.pca_model_path )
            test_vec_pca = pca.transform(test_vec_scala) 
    print("inputText:  " + text)
    if not os.path.exists(config.CLUSTER_MODEL):
        raise ValueError("%s model is not exist,please run dl_clustering first" % config.cluster_type)
    else:
        if config.cluster_type == 'kmeans':
            current_model = tc.load_model(config.CLUSTER_MODEL)
            text_dataFrame = pd.DataFrame(test_vec_pca) ##[tc.SArray(i) for i in encode_vec[config.depth - 1].reshape(-1, 1).tolist()]
            text_SFrame = tc.SFrame(text_dataFrame)
            predict_ids = current_model.predict(text_SFrame).to_numpy()
            print("predicts cluster id:" + str(predict_ids[0]))

        elif config.cluster_type == "agglomerative":
            #raise ValueError("agglomerative  predict not impletion")
            model = joblib.load(config.CLUSTER_MODEL)
            train_df = tc.SFrame.read_csv(config.TRAIN_DATA_all, header=False, skiprows=None)
            train_dataFrame = train_df.to_dataframe()
            current_model = get_model(config.cluster_type, model.n_clusters,train_dataFrame)
            cluster_id = current_model.fit_predict(test_vec_pca)
            print("predicts cluster id:" + str(cluster_id[0]))

        else:
            model = joblib.load(config.CLUSTER_MODEL)
            try:
                cluster_id = model.predict(test_vec_pca)
            except Exception as e:
                cluster_id = model.fit_predict(test_vec_pca)

            print("predicts cluster id:" + str(cluster_id[0]))

if __name__ == '__main__':
    inputText = 'When power is pressed  Laptop goes through start up '
    predict_clusters_id(inputText)
