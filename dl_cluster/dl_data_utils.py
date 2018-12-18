# -*- coding: utf-8 -*-
import os
import re
import sys
import codecs
import dl_settings as config
import numpy as np
import codecs
from sklearn.model_selection import train_test_split

_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Remove symbols
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")

def basic_tokenizer(sentence):
    words = sentence.strip().split(" ")
    return [w for w in words if re.split(_WORD_SPLIT,w)]

# Generate word vacabulary
def create_vocabulary(data):
    vocab = {}
    count = 0
    for line in data:
        count +=1
        if count % 10000 == 0:
            print("Processed data：%d" % count)

        tokens = line.split(" ")
        for word in tokens:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab = {key: value for key, value in vocab.items() if value > config.filter_words_frequent}
    vocab_list = _START_VOCAB + sorted(vocab,key=vocab.get,reverse=True)
    with open(config.vocabulary,'w',encoding='utf-8') as vocab_write:
        for w in vocab_list:
            vocab_write.write(w + "\n")

#generate vacab list in vocabulary_path
def initialize_vocabulary(vocabulary_path):
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs.open(vocabulary_path,'r', encoding = "utf8") as f:
            rev_vocab.extend(f.readlines())

        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict((x,y) for (y,x) in enumerate(rev_vocab))
        return vocab,rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found",vocabulary_path)

# Map a sentence into id according to vocab
def sentence_to_token_ids(sentence,vocabulary):
    words = sentence.split(" ")
    return [vocabulary.get(w,UNK_ID) for w in words]

# Map sentences in data_path to sentence id, according to vocab and generate new files
def data_to_token_ids(data, target_path, valid_path,vocabulary_path):
    vocab,_ = initialize_vocabulary(vocabulary_path)
    totol_data = []
    count = 0
    for line in data:
        count +=1
        if count % 10000 == 0:
            print("Already processed lines of %d" % count)
        token_ids = sentence_to_token_ids(line,vocab)
        totol_data.append(token_ids)

    # Divide into train and test dataset
    test_num = int(len(totol_data) * 0.1)
    train_data,test_data = totol_data[test_num:],totol_data[:test_num]
    with open(target_path, 'w', encoding='utf-8') as tokens_file:
        for token_ids in train_data:
            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    with open(valid_path, 'w', encoding='utf-8') as tokens_file:
        for token_ids in test_data:
            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def read_data(tokenized_data_path):
    data_set = dict()
    encode =[]
    encode_length = []

    with codecs.open(tokenized_data_path,'r',encoding='utf-8') as fh:
        for line in fh:
            counter = 0
            counter += 1
            if counter % 10000 == 0:
                print("Already read lines of %d" % counter)

            source_ids = [int(x) for x in line.split(" ")]
            encode_length.append(len(source_ids))
            encode.append(source_ids)
    data_set['encode'] = np.asarray(encode)
    data_set['encode_lengths'] = np.asarray(encode_length)
    return data_set

class BatchManager():
    def __init__(self,data_path,batch_size):
        data_set = read_data(data_path)
        self.encode = data_set['encode']
        self.encode_lengths = data_set['encode_lengths']
        self.total = len(self.encode)
        self.batch_size = batch_size
        self.num_batch = int(self.total/self.batch_size)

    def shuffle(self):
        shuffle_index = np.random.permutation(np.arange(self.total))
        self.encode = self.encode[shuffle_index]
        self.encode_lengths = self.encode_lengths[shuffle_index]

    def pad_sentence_batch(self,sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        if max_sentence > config.max_seq_length:
            max_sentence = config.max_seq_length
        sentence_clean = []
        sentence_len = []
        for sentence in sentence_batch:
            if len(sentence) > max_sentence:
                sentence_len.append( max_sentence)
                sentence_clean.append(sentence[0:max_sentence])
            else:
                sentence_len.append(len(sentence))
                sentence_clean.append( sentence + [pad_int] * (max_sentence - len(sentence)))
        return ( np.array(sentence_clean), np.array(sentence_len) )

    def iterbatch(self,shuffle=True):
        if shuffle:
            self.shuffle()
        for i in range(self.num_batch + 1):
            if i == self.num_batch:
                if self.encode[i * self.batch_size:].tolist():
                    encode,encode_length  = self.pad_sentence_batch(self.encode[i * self.batch_size:].tolist(),PAD_ID)
                else:
                    encode, encode_length = np.array([]), np.array([])
                data = {"encode": encode, "encode_lengths": encode_length}
            else:
                encode,encode_length = self.pad_sentence_batch(self.encode[i * self.batch_size:(i + 1) * self.batch_size].tolist(), PAD_ID)
                data = {"encode": encode, "encode_lengths": encode_length}
            yield data
