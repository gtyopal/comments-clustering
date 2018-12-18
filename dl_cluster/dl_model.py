# -*- coding: utf-8 -*-
import os
import sys
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn, seq2seq
import dl_data_utils as data_utils
import math

class Seq2SeqModel(object):
    # reference code : https: // github.com / JayParks / tf - seq2seq
    def __init__(self, config, model, vocabularySize):
        assert model.lower() in ['train','decode']
        self.mode = model.lower()
        self._init_config(config, vocabularySize)
        self._init_placeholders()
        self._init_embeddings()
        self._build_network()
        if self.mode.lower() == "train":
            self._init_optimizer()
        # Merge all the training summaries
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=3)

    def _init_config(self, config, vocabularySize):
        self.config = config
        self.cell_type = config.cell_type
        self.hidden_units = config.hidden_units
        self.depth = config.depth
        self.attention_type = config.attention_type
        self.embedding_size = config.embedding_size
        self.num_encoder_symbols = vocabularySize
        self.num_decoder_symbols = vocabularySize
        self.use_residual = config.use_residual
        self.attn_input_feeding = config.attn_input_feeding
        self.use_dropout = config.use_dropout
        self.keep_prob = 1.0 - config.dropout_rate
        self.optimizer = config.optimizer
        self.learning_rate = config.learning_rate
        self.max_gradient_norm = config.max_gradient_norm
        self.global_step = tf.Variable(0,trainable=False,name='global_step')
        self.global_epoch_step = tf.Variable(0,trainable=False,name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        lr = tf.train.exponential_decay(0.001, global_step=self.global_step,
                                        decay_steps= 500, decay_rate=0.8,
                                        staircase=True)
        self.learning_rate= tf.maximum(lr, 0.0001)
        self.dtype = tf.float16 if config.use_fp16 else tf.float32
        self.keep_prob_placeholder = tf.placeholder(self.dtype, shape=[], name='keep_prob')
        self.mode_save_path = os.path.join(config.model_dir,config.model_name)
        self.use_beamsearch_decode = False


    def _init_placeholders(self):
        # encode_inputs:[batch_size, max_time_steps]
        self.encoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None, None), name='encoder_inputs')

        #encode_inputs_length [batch_size]
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,), name='encoder_inputs_length')

        # get dynamic batch_size
        self.batch_size = tf.shape(self.encoder_inputs)[0]
        if self.mode == 'train':
            # decoder_inputs:[batch_size,max_time_steps]
            self.decoder_inputs = tf.placeholder(dtype=tf.int32,shape=(None, None), name='decoder_inputs')
            # decoder_inputs_length: [batch_size]
            self.decoder_inputs_length = tf.placeholder(dtype=tf.int32,shape=(None,), name='decoder_inputs_length')

            decoder_start_token = tf.ones(shape=[self.batch_size,1], dtype=tf.int32) * data_utils.GO_ID
            decoder_end_token = tf.ones(shape=[self.batch_size,1], dtype=tf.int32) * data_utils.EOS_ID

            # decoder_inputs_train: [batch_size , max_time_steps + 1]
            # insert _GO symbol in front of each decoder input
            self.decoder_inputs_train = tf.concat([decoder_start_token,
                                                   self.decoder_inputs], axis=1)

            # decoder_inputs_length_train: [batch_size]
            self.decoder_inputs_length_train = self.decoder_inputs_length + 1

            # decoder_targets_train: [batch_size, max_time_steps + 1]
            # insert EOS symbol at the end of each decoder input
            self.decoder_targets_train = tf.concat([self.decoder_inputs,
                                                    decoder_end_token], axis=1)


    def _init_embeddings(self):
        with tf.name_scope('word_embedding'):
            # encode and decode share one embedding
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=tf.float32)
            self.embeddings = tf.get_variable(name='embedding_matrix',shape=[self.num_encoder_symbols,self.embedding_size],dtype=tf.float32,initializer=initializer)


    def build_single_cell(self,layer=1):
        cell_type = rnn.LSTMCell
        if (self.cell_type.lower() == 'gru'):
            cell_type = rnn.GRUCell
        cell = cell_type(self.hidden_units * layer)

        if self.use_dropout:
            cell = rnn.DropoutWrapper(cell,dtype=self.dtype,output_keep_prob=self.keep_prob_placeholder)

        if self.use_residual:
            cell = rnn.ResidualWrapper(cell)
        return cell

    def build_encode(self):
        with tf.variable_scope('encoder'):
            # building encoder_cell
            self.encoder_cell = rnn.MultiRNNCell([self.build_single_cell() for _ in range(self.depth)])

            # Embedded_inputs: [batch_size, time_step, embedding_size]
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings,self.encoder_inputs)

            # Input projection layer to feed embedded inputs to the cell
            input_layer = Dense(self.hidden_units,dtype=self.dtype,name='input_projection')

            # Embedded inputs having gone through input projection layer
            self.encoder_inputs_embedded = input_layer(self.encoder_inputs_embedded)

            # Convert input sentence into context vector
            # encoder_outputs: [batch_size, max_time_step, cell_output_size]
            # encoder_state: [batch_size, cell_output_size]
            self.encoder_outputs,self.encoder_last_state = tf.nn.bidirectional_dynamic_rnn(
                rnn.MultiRNNCell([self.build_single_cell() for _ in range(self.depth)]),
                rnn.MultiRNNCell([self.build_single_cell() for _ in range(self.depth)]),
                inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=self.dtype,
                time_major=False
            )
            self.encoder_outputs = tf.concat(self.encoder_outputs, 2)
            encoder_states = []

            for i in range(self.depth):
                if isinstance(self.encoder_last_state[0][i], tf.contrib.rnn.LSTMStateTuple):
                    encoder_state_c = tf.concat(values=(self.encoder_last_state[0][i].c, self.encoder_last_state[0][i].c), axis=1,
                                                name="encoder_fw_state_c")
                    encoder_state_h = tf.concat(values=(self.encoder_last_state[0][i].h, self.encoder_last_state[0][i].h), axis=1,
                                                name="encoder_fw_state_h")
                    encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                elif isinstance(self.encoder_last_state[0][i], tf.Tensor):
                    encoder_state = tf.concat(values=(self.encoder_last_state[0][i], self.encoder_last_state[1][i]), axis=1,
                                              name='bidirectional_concat')

                encoder_states.append(encoder_state)

            self.encoder_last_state = tuple(encoder_states)


    def build_decode_cell(self):
        encoder_outputs = self.encoder_outputs
        encoder_last_state = self.encoder_last_state
        encoder_inputs_length = self.encoder_inputs_length
        # Building attention mechanism: Default Bahdanau
        # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
        self.attention_mechanism = seq2seq.BahdanauAttention(
            num_units=self.hidden_units,
            memory=encoder_outputs,
            memory_sequence_length=encoder_inputs_length
        )

        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = seq2seq.LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_inputs_length
            )

        # decoder_cell
        self.decoder_cell_list = [self.build_single_cell(layer=2) for _ in range(self.depth)]

        def attn_decoder_input_fn(inputs, attention):
            if not self.attn_input_feeding:
                return inputs
            # Essential when use_residual=True
            _input_layer = Dense(self.hidden_units * 2, dtype=self.dtype, name='attn_input_feeding')
            return _input_layer(tf.concat([inputs, attention], -1))

        # AttentionWrapper wraps RNNCell with the attention_mechanism
        # Note: We implement Attention mechanism only on the top decoder layer
        self.decoder_cell_list[-1] = seq2seq.AttentionWrapper(
            cell=self.decoder_cell_list[-1],
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            cell_input_fn=attn_decoder_input_fn,
            initial_cell_state=encoder_last_state[-1],
            alignment_history=False,
            name='Attention_wrapper'
        )

        # To be compatible with AttentionWrapper, the encoder last state
        # of the top layer should be converted into the AttentionWrapperState form
        # We can easily do this by calling AttentionWrapper.zero_state

        batch_size = self.batch_size
        initial_state = [state for state in encoder_last_state]
        initial_state[-1] = self.decoder_cell_list[-1].zero_state(
            batch_size=batch_size, dtype=self.dtype
        )
        decoder_initial_state = tuple(initial_state)
        return rnn.MultiRNNCell(self.decoder_cell_list), decoder_initial_state


    def build_decode(self):
        # build decoder and attention.
        with tf.variable_scope('decoder'):
            self.decoder_cell,self.decoder_initial_state = self.build_decode_cell()

            # Input projection layer to feed embedded inputs to the cell
            # ** Essential when use_residual=True to match input/output dims
            input_layer = Dense(self.hidden_units*2,dtype=self.dtype,name='input_projection')

            # Output projection layer to convert cell_outpus to logits
            output_layer = Dense(self.num_decoder_symbols,name='output_project')

            if self.mode == 'train':
                # decoder_inputs_embedded: [batch_size, max_time_step + 1, embedding_size]
                self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs_train)

                # Embedded inputs having gone through input projection layer
                self.decoder_inputs_embedded = input_layer(self.decoder_inputs_embedded)

                # Helper to feed inputs for training: read inputs from dense ground truth vectors
                training_helper = seq2seq.TrainingHelper(inputs=self.decoder_inputs_embedded,
                                                         sequence_length=self.decoder_inputs_length_train,
                                                         time_major=False,
                                                         name='training_helper')

                training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                        helper=training_helper,
                                                        initial_state=self.decoder_initial_state,
                                                        output_layer=output_layer)


                #Maximum decoder time_steps in current batch
                max_decoder_length = tf.reduce_max(self.decoder_inputs_length_train)

                (self.decoder_outputs_train,self.decoder_last_state_train,self.decoder_outputs_length_train) = (seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                                                                       output_time_major=False,
                                                                                                                                       impute_finished=True,
                                                                                                                                       maximum_iterations=max_decoder_length))
                # More efficient to do the projection on the batch-time-concatenated tensor
                # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
                # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
                self.decoder_logits_train = tf.identity(self.decoder_outputs_train.rnn_output)

                # Use argmax to extract decoder symbols to emit
                self.decoder_pred_train = tf.argmax(self.decoder_logits_train,axis=-1,
                                                    name='decoder_pre_train')

                # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
                masks = tf.sequence_mask(lengths=self.decoder_inputs_length_train,
                                         maxlen=max_decoder_length,
                                         dtype=self.dtype,
                                         name='masks')

                self.loss = seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                  targets=self.decoder_targets_train,
                                                  weights=masks,
                                                  average_across_timesteps=True,
                                                  average_across_batch=True)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
            elif self.mode == 'decode':
                # Start_tokens: [batch_size,] `int32` vector
                start_token = tf.ones([self.batch_size,],tf.int32) * data_utils.GO_ID
                end_token = data_utils.EOS_ID

                def embed_and_input_proj(inputs):
                    return input_layer(tf.nn.embedding_lookup(self.embeddings,inputs))


                decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_token,end_token=end_token,
                                                                embedding=embed_and_input_proj)
                inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                                                         helper=decoding_helper,
                                                         initial_state=self.decoder_initial_state,
                                                         output_layer=output_layer)


                (self.decoder_outputs_decode,self.decoder_last_state_decode,self.decoder_outputs_length_decode) = (seq2seq.dynamic_decode(decoder=inference_decoder,
                                        output_time_major=False,
                                        maximum_iterations=self.config.max_decode_step))


                self.decoder_pred_decode = tf.expand_dims(self.decoder_outputs_decode.sample_id, -1)

        # Building decoder cell and attention. Also returns decoder_initial_state

    def _build_network(self):
        # build encoder networks
        self.build_encode()

        # build decoder networks
        self.build_decode()

    def _init_optimizer(self):
        with tf.name_scope("optimizer"):
            train_params = tf.trainable_variables()
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.loss,train_params)

            # Clip_gradients by a given maximum_gradient_norm
            clip_gradients,_ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)

            # Update the model
            self.updates = self.opt.apply_gradients(zip(clip_gradients,train_params),global_step=self.global_step)


    # Generate feed_dict
    def make_feeds_dict(self, encoder_inputs, encoder_inputs_length,
                    decoder_inputs, decoder_inputs_length, decode):
        """
        Args:
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decode: a scalar boolean that indicates decode mode
        Returns:
          A feed for the model that consists of encoder_inputs, encoder_inputs_length,
          decoder_inputs, decoder_inputs_length
        """
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError("Encode inputs don't match with its length,%d != %d" % (input_batch_size,encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError("Encode inputs don't match with decode inputs,%d != %d" % (input_batch_size,target_batch_size))

            if target_batch_size != decoder_inputs_length.shape[0]:
                raise ValueError("Decoder targets don't match with its length, %d != %d" % (target_batch_size, decoder_inputs_length.shape[0]))


        feed_dict = {}
        feed_dict[self.encoder_inputs.name] = encoder_inputs
        feed_dict[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            feed_dict[self.decoder_inputs.name] = decoder_inputs
            feed_dict[self.decoder_inputs_length.name] = decoder_inputs_length

        return feed_dict

    #
    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        """Run a train step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        if self.mode.lower() != 'train':
            raise ValueError("train step can only be operated in train mode")

        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          decoder_inputs,decoder_inputs_length,False)
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        output_feed = [self.updates,self.loss,self.summary_op]
        _,loss,summary_op = sess.run(output_feed,input_feed)
        return loss,summary_op

    # Evaluate model
    def eval(self, sess, encoder_inputs, encoder_inputs_length,
             decoder_inputs, decoder_inputs_length):
        """Run a evaluation step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
              to feed as encoder inputs
          encoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch
          decoder_inputs: a numpy int matrix of [batch_size, max_target_time_steps]
              to feed as decoder inputs
          decoder_inputs_length: a numpy int vector of [batch_size]
              to feed as sequence lengths for each element in the given batch

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.
        """
        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          decoder_inputs,decoder_inputs_length,False)
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = [self.loss,self.summary_op]
        loss,summary_op = sess.run(output_feed,input_feed)
        return loss,summary_op


    #Predict tags for input sentence
    def predict(self,sess,encoder_inputs,encoder_inputs_length,vocab_list):
        input_feed = self.make_feeds_dict(encoder_inputs,encoder_inputs_length,
                                          None,None,True)
        input_feed[self.keep_prob_placeholder.name] = 1.0
        output_feed = self.decoder_pred_decode
        for key in input_feed.keys():
            print(key)
        print(self.decoder_pred_decode.name)
        predicts = sess.run(output_feed,input_feed)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        print(predicts.shape)
        for token in predicts[0]:
            selected_token_id = int(token)
            if selected_token_id == data_utils.EOS_ID or selected_token_id == data_utils.PAD_ID:
                break
            else:
                outputs.append(selected_token_id)

        # Forming output sentence on natural language
        output_sentence = " ".join([vocab_list[output] for output in outputs])
        return output_sentence

    def seq2words(self,seq, inverse_target_dictionary):
        words = []
        for w in seq:
            if w == data_utils.EOS_ID or w == data_utils.PAD_ID:
                break
            words.append(inverse_target_dictionary[w])
        return ' '.join(words)


    # Restore or initiate model
    def model_restore(self,sess):
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.mode_save_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restore model...")
            self.saver.restore(sess,ckpt.model_checkpoint_path)

        else:
            print("Initiate model...")
            sess.run(tf.global_variables_initializer())

