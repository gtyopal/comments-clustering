# coding: utf-8
import os
PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
data_dir = os.path.join(PARENT_DIR_PATH,'data')
model_dir = os.path.join(PARENT_DIR_PATH, "checkpoint",'model')
LOG_FILE = os.path.join(PARENT_DIR_PATH, 'logs', 'cluster')
RAW_DATA = os.path.join(PARENT_DIR_PATH, 'clusterdata','feedback-sample.txt')
TRAIN_DATA_CLEAN  = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_clean.csv')
TRAIN_DATA_word2vec = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_word.csv')
TRAIN_DATA_lsi = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_lsi.csv')
TRAIN_DATA_seq2seq = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_seq2seq.csv')
TRAIN_DATA_all = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_all.csv')
TRAIN_DATA_all_2d = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train_data_all_2D.csv')
dictionary_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'all.dictionary')
tfidf_model_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'tfidf.model')
lsi_model_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'lsi.model')
word2vec_model_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'word2vec.model')
scalar_model_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'scalar.model')
pca_model_path = os.path.join(PARENT_DIR_PATH,"checkpoint",'pca.model')

cluster_type = 'agglomerative'
if cluster_type == 'minibatchkmeans':
    CLUSTER_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoint", 'minibatchkmeans')
elif cluster_type == 'hdbscan':
    CLUSTER_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoint", 'hdbscan')
elif cluster_type == 'agglomerative':
    CLUSTER_MODEL = os.path.join(PARENT_DIR_PATH, "checkpoint", 'agglomerative')
cluster_tag_files = os.path.join(PARENT_DIR_PATH,'clusterdata', cluster_type + '_cluster_tags.csv')
cluster_tag_files_clusterid2name = os.path.join(PARENT_DIR_PATH,'clusterdata', cluster_type + '_cluster_tags_with_id2name_res.csv')
cluster_tag_files_clusterid2name_tmp = os.path.join(PARENT_DIR_PATH,'clusterdata', cluster_type + '_cluster_tags_with_id2name_tmp.txt')
# clustering model parameters
sentence_max_len = 500
cluster_num = 10
min_num_members_per_cluster = 2
max_num_cluster = 30

# training dataset parameters
vocabulary  = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'vocab.txt')
train_file  = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'train.txt')
valid_file  = os.path.join(PARENT_DIR_PATH, 'clusterdata', 'valid.txt')
vocabulary_size = 100000
filter_words_frequent = 2
filter_sentence_len = 2

# tensorflow parameters
cell_type = 'gru'              # RNN cell type ,default: lstm
attention_type = 'bahdanau'    # attentions mechanism type: (bahdanau, luong), default: bahdanau
hidden_units = 100             # number of hidden units
depth = 2                       # number of neural network layers
embedding_size = 100           # Embedding dimensions
use_residual = True            # if or not use residual network between layers
attn_input_feeding = True      # Use input feeding method in attentional decoder
use_dropout = True             # if or not use dropout on rnn cell
dropout_rate = 0.5              # Dropout probability for input/output/state units (0.0: no dropout)')

# training parameters
max_epochs = 100                # number of max epochs
learning_rate = 0.001          # learning rate
split_rate = 0.01              # split rate
max_gradient_norm = 3.0        # gradient truncate value
batch_size = 64                 # Batch size
max_load_batches = 20           # max number of loading training batches
max_seq_length = 100           # max length of sentences
display_freq = 10              # number of steps to trained to display result
save_freq = 1150                # number of steps to trained to save result
valid_freq = 20                # number of steps to trained to validate model
optimizer = 'adam'              # optimizer: (adadelta, adam, rmsprop)
model_name = 'cluster.ckpt'        # name of saved model
shuffle_each_epoch = True       # if or not to shuffle train dataset after each epoch
sort_by_length = True           # if or not to sort train dataset on length of sentence
use_fp16 = False               # Use half precision float16 instead of float32 as dtype
max_decode_step = 10
w2v_dim = 300                  # word2vec dimension
lsi_dim = 100                  # lsi dimension
pca_dim = 256                  # pca dimension