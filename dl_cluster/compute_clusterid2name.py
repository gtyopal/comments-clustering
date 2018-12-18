# coding: utf-8
import os
import sys
PROJECT_ROOT =  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import pandas as pd
import dl_settings as config
from  dl_cluster.data_preprocessing import denoise_text
from collections import defaultdict
import math
import codecs 

data_path = config.cluster_tag_files
df = pd.read_csv(data_path )
df["message_clean"] = df["message"].apply( denoise_text )
df["message_clean"] = df["message_clean"].apply( lambda x : x.strip().split() )
cluster_id = df["cluster_id"].tolist()
message_clean = df["message_clean"].tolist()

doc_count = len( cluster_id )
# compute the base count 
token_doc_dict = defaultdict(int)
label_doc_dict = defaultdict(int)
for mess_tmp, label_tmp in zip(message_clean,  cluster_id ):
    word_set = set(  mess_tmp )
    for token in word_set:
        token_doc_dict[ token ] += 1
    label_doc_dict[ label_tmp ] += 1

token_doc_dict_per_label = {}
for label in set( cluster_id ):
    df_tmp = df[ df["cluster_id"]== label]
    mess_cut = df_tmp["message_clean"].tolist()
    token_dict_tmp = defaultdict(int)
    for mess_cut_tmp in mess_cut:
        for token in set(mess_cut_tmp):
            token_dict_tmp[token] += 1
    token_doc_dict_per_label[ label ] = token_dict_tmp

# compute score with f1 function
token_score_dict_per_label = {}
for label in set( cluster_id ):
    token_doc_dict_tmp = token_doc_dict_per_label[label]
    label_count = label_doc_dict[label]
    token_score_dict_tmp = {}
    for token,count in token_doc_dict_tmp.items():
        token_score_dict_tmp[ token ] = []
        precision_tmp = count* 1.0 / token_doc_dict[token]
        recall_tmp = count* 1.0 / label_count
        f1_tmp = 2* (precision_tmp * recall_tmp) / ( precision_tmp + recall_tmp )
        token_score_dict_tmp[token].extend( [precision_tmp, recall_tmp, f1_tmp] )
    token_score_topN = sorted( token_score_dict_tmp.items(), key= lambda item: item[1][2], reverse=True )[:10]
    token_score_dict_per_label[ label ] = token_score_topN

# compute score with information gain
token_ig_score_dict_per_label = {}
for label in set( cluster_id ):
    token_doc_dict_tmp = token_doc_dict_per_label[label]
    label_count = label_doc_dict[label]
    token_ig_score_dict_tmp = {}
    for token,count in token_doc_dict_tmp.items():
        tmp1 = count*1.0/( token_doc_dict[token])
        tmp1_rate = token_doc_dict[token]* 1.0 / doc_count
        tmp_count = doc_count - label_count - ( token_doc_dict[token] -count)
        if tmp_count != 0:
            tmp2 = tmp_count*1.0 /  ( doc_count - token_doc_dict[token] )
        else:
            tmp2 = 0
        tmp2_rate = ( doc_count - token_doc_dict[token] )* 1.0 / doc_count 
        ig_score = tmp1_rate * tmp1 * math.log( tmp1 + 1e-10) + tmp2_rate * tmp2 * math.log( tmp2 + 1e-10)
        token_ig_score_dict_tmp[token] =  ig_score
    token_score_topN = sorted( token_ig_score_dict_tmp.items(), key= lambda item: item[1], reverse=True )[:10]
    token_ig_score_dict_per_label[ label ] = token_score_topN

# reranking with the two scores
token_score_merge_per_label = {}
for label in set(cluster_id):
    label_tmp1 = token_score_dict_per_label[ label ]
    label_tmp2 = token_ig_score_dict_per_label[ label ]
    token_score_dict_tmp = defaultdict( float)
    for i in range( len(label_tmp1)):
        
        token_score_dict_tmp[ label_tmp1[i][0] ] +=  2.0 - i * 0.1
    for i in range( len(label_tmp2)):
        token_score_dict_tmp[ label_tmp2[i][0] ] +=  2.0 - i * 0.1
    token_score_merge = sorted( token_score_dict_tmp.items(), key= lambda item: item[1], reverse=True )
    token_score_merge = [ (sco[0], round(sco[1],3)) for sco in token_score_merge ]  
    token_score_merge_per_label[ label ] = token_score_merge

f1 = codecs.open( config.cluster_tag_files_clusterid2name_tmp, "w", encoding = "utf8")
f1.write("cluster_id" + " : " +  "candidate word" + "\n")
for k,v in token_score_merge_per_label.items():
    f1.write( str(k) + " : " + str(v) + "\n")
f1.close()

label_res = {}
for idx, token_res in token_score_merge_per_label.items():
    token_tmp = []
    for tok in token_res:
        if not tok[0].isdigit():
            token_tmp.append( tok[0] )
        if len( token_tmp ) == 2:
            break
    label_res[ idx ] = ",".join( token_tmp )

df["represent_name"] = df["cluster_id"].apply( lambda idx : label_res[ idx ])
df.to_csv(config.cluster_tag_files_clusterid2name, index = False , encoding = "utf8")