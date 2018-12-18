# coding: utf-8
import os
import sys
import tensorflow as tf
import dl_settings as config
import dl_model as model
import dl_data_utils as data_utils
import data_preprocessing as dp
from sklearn.preprocessing import StandardScaler
import time
import math
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def train():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        dp.prepare_data_for_model()
        print("Preprocessed data...")
        tran_batch_manager = data_utils.BatchManager(config.train_file, config.batch_size)
        valid_batch_manager = data_utils.BatchManager(config.valid_file, config.batch_size)
        _,vocab_list = data_utils.initialize_vocabulary(config.vocabulary)
        with tf.device('/gpu:0'):
            model_obj = model.Seq2SeqModel(config, 'train', len(vocab_list))
            model_obj.model_restore(sess)
            print("Start training seq2seq model...")
            loss = 0.0
            start_time = time.time()
            best_loss = 10000.0
            total_vec = []
            for epoch_id in range(config.max_epochs):
                print('Epoch {}'.format(epoch_id+1))
                for step, train_batch in enumerate(tran_batch_manager.iterbatch()):
                    if train_batch['encode'] is None or not train_batch['encode'].tolist():
                        continue
                    # Execute a single training step
                    try:
                        step_loss, summary = model_obj.train(sess, encoder_inputs=train_batch['encode'],
                                                                     decoder_inputs= train_batch['encode'],
                                                                     encoder_inputs_length=train_batch['encode_lengths'],
                                                                     decoder_inputs_length = train_batch['encode_lengths'])
                        loss += float(step_loss) / config.display_freq
                    except Exception as e:
                        print('Training error...')
                        continue


                    if (model_obj.global_step.eval()+1) % config.display_freq == 0:
                        if (loss < best_loss) and ((model_obj.global_step.eval() +1) % (config.display_freq+45) == 0):
                            best_loss = loss
                            print("Save model...")
                            checkpoint_path = model_obj.mode_save_path
                            model_obj.saver.save(sess, checkpoint_path, global_step=model_obj.global_step)

                        avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                        #Time calculation
                        time_cost = time.time() - start_time
                        step_time = time_cost / config.display_freq
                        print('Train Epoch of %d，Step of %d，Loss of %.2f , Preplexity of %.2f, Time of %f' % (epoch_id +1, model_obj.global_step.eval(),loss,avg_perplexity,step_time))
                        loss = 0.0
                        start_time = time.time()

                    if (model_obj.global_step.eval() + 1) % config.valid_freq == 0:
                        valid_loss = 0.0
                        total_sentence = 0
                        for test_batch in valid_batch_manager.iterbatch():
                            step_loss, summary = model_obj.eval(sess, encoder_inputs=test_batch['encode'],
                                                                decoder_inputs=test_batch['encode'],
                                                                encoder_inputs_length=test_batch['encode_lengths'],
                                                                decoder_inputs_length=test_batch['encode_lengths'])
                            batch_size = test_batch['encode_lengths'].shape[0]
                            valid_loss += step_loss * batch_size
                            total_sentence += batch_size
                        valid_loss = valid_loss / total_sentence
                        print("     Validate: Validation Loss of %.2f, Validation Preplexity of %.2f" % (valid_loss, math.exp(valid_loss)))

                    if (model_obj.global_step.eval()+1) % config.save_freq ==0:
                        print("Save model...")
                        checkpoint_path = model_obj.mode_save_path
                        model_obj.saver.save(sess, checkpoint_path, global_step=model_obj.global_step)

            print("Completed seq2seq training...")

            tran_batch_manager = data_utils.BatchManager(config.train_file, config.batch_size)
            for step, tran_batch in enumerate(tran_batch_manager.iterbatch(shuffle=False)):
                if tran_batch['encode'] is None or not tran_batch['encode'].tolist():
                    continue
                feed_dict = {}
                feed_dict[model_obj.encoder_inputs.name] = tran_batch['encode']
                feed_dict[model_obj.encoder_inputs_length.name] = tran_batch['encode_lengths']
                feed_dict[model_obj.keep_prob_placeholder.name] = 1.0
                [encoder_vec] = sess.run([model_obj.encoder_last_state], feed_dict=feed_dict)
                total_vec.append(encoder_vec[1].tolist())

            valid_batch_manager = data_utils.BatchManager(config.valid_file, config.batch_size)
            for step, tran_batch in enumerate(valid_batch_manager.iterbatch(shuffle=False)):
                if tran_batch['encode'] is None or not tran_batch['encode'].tolist():
                    continue
                feed_dict = {}
                feed_dict[model_obj.encoder_inputs.name] = tran_batch['encode']
                feed_dict[model_obj.encoder_inputs_length.name] = tran_batch['encode_lengths']
                feed_dict[model_obj.keep_prob_placeholder.name] = 1.0
                [encoder_vec] = sess.run([model_obj.encoder_last_state], feed_dict=feed_dict)
                total_vec.append(encoder_vec[1].tolist())

            seq2seq_vecr_series = np.concatenate(total_vec, axis=0)
            df_seq2seq = pd.DataFrame(seq2seq_vecr_series)
            print("  seq2seq_vec shape: ", df_seq2seq.shape)
            print("make train_data with seq2seq done!")
            return df_seq2seq


def merge_all_features():
    df_seq2seq= train()
    df_lsi, df_w2v = dp.get_lsi_word2vec_vec()
    word2vec_matrix = dp.get_stats(df_w2v)
    lsi_matrix = dp.get_stats(df_lsi)
    seq2seq_matrix = dp.get_stats(df_seq2seq)
    print("Start to merge train vectors...")
    vec_all = np.hstack((word2vec_matrix,lsi_matrix,seq2seq_matrix))
    # Vectors scaling and pca dimensionality reduction
    scalar = StandardScaler().fit(vec_all)
    vec_scala = scalar.transform(vec_all)
    pca = PCA(n_components=config.pca_dim)
    pca = pca.fit(vec_scala)
    vec_pca = pca.transform(vec_scala)
    print("  merged_vec shape: ", vec_pca.shape)
    pd.DataFrame(vec_pca).to_csv(config.TRAIN_DATA_all, index=False, header=False)
    print("Merged features done!")
    print("Start converting train data to 2D dimension")
    tsne = TSNE(n_components=2)
    data_2D = tsne.fit_transform(pd.DataFrame(vec_pca))
    data_2D.tofile(config.TRAIN_DATA_all_2d)
    print("train data to vec and 2d done!")

if __name__ == '__main__':
    merge_all_features()
