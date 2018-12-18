# coding: utf-8
import os
import sys
import re
import string
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.stem import  WordNetLemmatizer
from nltk.tokenize import word_tokenize
import dl_settings as config
from gensim import corpora, models
from nltk.corpus import stopwords as pw
from scipy.sparse import csr_matrix
from gensim.models import Word2Vec
import dl_data_utils as data_utils
import csv


def remove_punctuation(text):
    text = re.sub(r'[^\x00-\x7f]',r' ',text)
    text = re.sub("["+string.punctuation+"]", " ", text)
    new_words = []
    words = word_tokenize(text)
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return " ".join(new_words)

def remove_non_ascii(text):
    """ remove non-ascii """
    words = word_tokenize(text)
    new_words = []
    for word in words:
        if re.findall(r'[^a-z0-9\,\.\?\:\;\"\'\[\]\{\}\=\+\-\_\)\(\^\&\$\%\#\@\!\`\~ ]', word):
            continue
        new_words.append(word)
    return " ".join(new_words)

def remove_others(text):
    """ remove url """
    text = re.sub(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?', ' url ' , text)
    """ remove email """
    text = re.sub(r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)', ' email ', text)
    """ remove phone numbers """
    text = re.sub(r'[\@\+\*].?[014789][0-9\+\-\.\~\(\) ]+.{6,}', ' phone ', text)
    """ remove digits """
    text = re.sub(r'[0-9\.\%]+', ' digit ', text)
    return text

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def remove_stop_words(words):
    return [ word for word in words if word not in pw.words("english")]

def denoise_text(text):
    try:
        text_ = text['text']
    except Exception as e:
        text_ = text
    text = text_.lower().strip()
    text = remove_punctuation(text)
    text = remove_non_ascii(text)
    words = word_tokenize(text)
    words = remove_stop_words(words)
    words = to_lowercase(words)
    words = lemmatize_verbs(words)
    text = " ".join(words)
    if not text.strip():
        text = " "
    return text

def removeUNK(text,vocab_list):
    try:
        text_ = text['text']
    except Exception as e:
        text_ = text

    word_list = text_.split(" ")
    word_list = [word for word in word_list if word in vocab_list]
    new_word_list = []
    for word in word_list:
        if word not in vocab_list:
            new_word_list.append(data_utils._UNK)
        else:
            new_word_list.append(word)
    if len(new_word_list) < config.filter_sentence_len:
        new_text = " "
    else:
        new_text = " ".join(word_list)
    if not new_text.strip():
        new_text = " "
    return new_text


def load_dataset():
    print("Reading CSV file.")
    pd_data = pd.read_csv(config.RAW_DATA, sep=",", encoding='utf-8', error_bad_lines=True, skip_blank_lines=True, header=None, delimiter="\t",
                          quoting=csv.QUOTE_NONE, engine='python')
    pd_data = pd_data.reset_index()
    pd_data.rename(columns={'index': 'row_id', 0: "message"}, inplace=True)
    pd_data = pd_data.dropna()
    pd_data["text"] = pd_data["message"]
    pd_data = pd_data.drop_duplicates(["text"])
    pd_data['text'] = pd_data['text'].apply(denoise_text)
    data_utils.create_vocabulary(pd_data['text'].tolist())
    vocab_dict, vocab_list = data_utils.initialize_vocabulary(config.vocabulary)
    pd_data['text'] = pd_data.apply(removeUNK, axis=1, args=(vocab_list,))
    pd_data = pd_data[pd_data["text"] != " "]
    print("clean_shape", pd_data.shape)
    pd_data.to_csv(config.TRAIN_DATA_CLEAN, index = False)
    return pd_data


def build_vectors_tfidf_lsi(train_data):
    # create dictionary
    dictionary = corpora.Dictionary()
    for line in train_data:
        dictionary.add_documents( [line.strip().split()] )
    small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < config.filter_words_frequent ]
    dictionary.filter_tokens(small_freq_ids)
    dictionary.compactify()
    dictionary.save(config.dictionary_path)
    print ("dictionary done!")

    # word2id
    corpus_bow  = []
    for line in train_data:
        word_bow = dictionary.doc2bow( line.strip().split() )
        corpus_bow.append(word_bow)
    # print(corpus_bow[:10])
    # create and save tfidf and lsi model
    tfidf_model = models.TfidfModel(corpus=corpus_bow, dictionary=dictionary)
    corpus_tfidf = [tfidf_model[doc] for doc in corpus_bow]
    # print(corpus_tfidf[:1])
    lsi_model = models.LsiModel(corpus = corpus_tfidf,
                                id2word = dictionary,
                                num_topics= config.lsi_dim)
    tfidf_model.save(config.tfidf_model_path)
    lsi_model.save(config.lsi_model_path)
    corpus_lsi = [lsi_model[doc] for doc in corpus_tfidf]
    # print(corpus_lsi[:1])

    data = []
    rows = []
    cols = []
    for line_count,line in enumerate(corpus_lsi):  #
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
    corpus_sparse_matrix = csr_matrix((data,(rows,cols))) # spare vector 
    corpus_matrix = corpus_sparse_matrix.toarray()  # dense vector
    # print(corpus_matrix[0])
    return dictionary, tfidf_model, lsi_model, corpus_matrix

def make_iter_article(articles):
    total = []
    for article in articles:
        total.append(article.strip().split())
    return total

def build_vectors_word2vec(articles):
    model = Word2Vec(make_iter_article(articles), size=config.w2v_dim, hs=1, sg=1, min_count=1, window=3, iter=10, negative=3, sample=0.001,  workers=4)
    model.save(config.word2vec_model_path)
    return model

# Prepare training data for model
def prepare_data_for_model():
    pd_data = load_dataset()
    train_data = pd_data['text'].tolist()
    data_utils.data_to_token_ids(train_data, config.train_file, config.valid_file, config.vocabulary)

def get_lsi_word2vec_vec():
    pd_data = pd.read_csv(config.TRAIN_DATA_CLEAN, sep=",", encoding='latin-1')
    train_data = pd_data['text'].tolist()

    #  create vec with lsi model
    dictionary, tfidf_model, lsi_model, corpus_matrix = build_vectors_tfidf_lsi(train_data)
    df_lsi_feature = corpus_matrix
    df_lsi = pd.DataFrame(df_lsi_feature)
    print("  lsi_vec shape: ", df_lsi.shape)
    print("Making train data with tf-idf-lsi done!")
    
    # create vec with word2vec model
    w2v_model = build_vectors_word2vec(train_data)
    data_cut = []
    data_vec = np.zeros((len(train_data),config.w2v_dim))
    for line in train_data:
        data_cut.append(line.strip().split())
    for sen_id, sen_cut in enumerate(data_cut):
        sen_vec = np.zeros(config.w2v_dim)
        count = 0
        for word in sen_cut:
            if word in w2v_model.wv:
                sen_vec += w2v_model.wv[word]
                count +=1
        if count != 0 :
            sen_vec = sen_vec / count
        data_vec[sen_id,:] = sen_vec
        df_w2v_feature = data_vec
    df_word2vec = pd.DataFrame(df_w2v_feature)
    print("  word2vec_vec shape: ", df_word2vec.shape)
    print("Make training data with word2vec done!")

    return df_lsi, df_word2vec

def get_stats(df):
    df["mean"] = df.mean(axis=1,skipna=True)
    df["var"] = df.var(axis=1,skipna=True)
    df["skew"] = df.skew(axis=1,skipna=True)
    df["kurt"] = df.kurt(axis=1,skipna=True)
    df_matrix = df.values
    return df_matrix

