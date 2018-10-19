from __future__ import division
import os
# import sys
# sys.path.append('../')
import time
from time import time
import tensorflow as tf
import numpy as np
from tqdm import *
import copy 
import logging
import json
from collections import OrderedDict

from colorama import init
from colorama import Fore, Back, Style

class SeqGAN():
    def __init__(self, sess, batch_size):

        ## graph
        self.sess = sess      

        self.batch_size = batch_size
        self.rnn_cell = 'gru'
        self.G_hidden_size = 512

        # ## dataset
        self.vocab_size = 26867
        self.image_feat_dim = 4096*3
        self.max_words = 70   # contain the <S> and </S>
        self.lstm_steps = self.max_words

        vocab_file = '../model/word2id_5.json'
        self.word2ix = json.load(open(vocab_file, 'r'))
        self.ix2word = {value: key for key,value in self.word2ix.items()}   

        self.START = self.word2ix['<S>']
        self.END = self.word2ix['</S>']
        self.UNK = self.word2ix['<UNK>']

        ## placeholder
        # image feat
        self.image_feat = tf.placeholder(tf.float32, [self.batch_size, self.image_feat_dim])
        self.sample_words_for_loss = tf.placeholder(tf.int32, [self.batch_size, self.max_words])

        # rollout_reward
        self.discounted_reward = tf.placeholder(tf.float32, [self.max_words*self.batch_size])
        

        ############################## Generator #############################
        sample_words_list = self.generator(name="G", reuse=False)
        sample_words_list_argmax = self.generator_test(name="G", reuse=True)

        sample_words = tf.stack(sample_words_list)
        self.sample_words = tf.transpose(sample_words, [1,0])    # B,S
        sample_words_argmax = tf.stack(sample_words_list_argmax)
        self.sample_words_argmax = tf.transpose(sample_words_argmax, [1,0])   # B,S

        ############################## Record parameters #####################
        load_G_params = tf.global_variables()
        params = tf.trainable_variables()
        self.load_G_params_dict = {}
        self.G_params = []
        self.G_params_dict = {}

        for param in load_G_params:
            if 'G' in param.name:
                self.load_G_params_dict.update({param.name: param})

        for param in params:
            if 'G' in param.name:
                self.G_params.append(param)
                self.G_params_dict.update({param.name: param})
        
        logging.info(Fore.GREEN + 'Build graph complete!')

    
    def batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_loss(self, name="generator", reuse=True):
        '''
           To compute loss.
           First feed in the sampled words to get the probabilities, then compute loss with 
           rewards got in advance.

           This function is added to correct the mistake caused by the twice 'sess.run' that 
           multinomial twice. 
        '''
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.device("/cpu:0"), tf.variable_scope("word"):
                # name: "gnerator/word"
                word_emb_W = tf.get_variable("word_emb_W", [self.vocab_size, self.G_hidden_size], tf.float32, random_uniform_init)
            
            with tf.variable_scope("image_feat"):
                # name: "generator/image_feat"
                image_feat_W = tf.get_variable("image_feat_W", [self.image_feat_dim, self.G_hidden_size], tf.float32, random_uniform_init)
                image_feat_b = tf.get_variable("image_feat_b", [self.G_hidden_size], tf.float32, random_uniform_init)
                
            with tf.variable_scope("output"):
                # name: "generator/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.vocab_size], tf.float32, random_uniform_init)
                output_b = tf.get_variable("output_b", [self.vocab_size], tf.float32, random_uniform_init)

            with tf.variable_scope("lstm_encoder"):
                if self.rnn_cell == 'lstm':
                    encoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    encoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)

            with tf.variable_scope("lstm_decoder"):
                # WONT BE CREATED HERE
                if self.rnn_cell == 'lstm':
                    decoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    decoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)
            
            #============================= dropout ===================================================================
            # to be confirmed
            if self.rnn_drop > 0:
                logging.debug(Fore.CYAN + 'using dropout in rnn!')
                encoder = tf.nn.rnn_cell.DropoutWrapper(encoder, input_keep_prob=1.0-self.rnn_drop, output_keep_prob=1.0)
                decoder = tf.nn.rnn_cell.DropoutWrapper(decoder, input_keep_prob=1.0, output_keep_prob=1.0-self.rnn_drop)

            #============================= encoder ===================================================================
            state = encoder.zero_state(self.batch_size, tf.float32)
            with tf.variable_scope("image_feat") as scope:
                image_feat = self.batch_norm(self.image_feat[:,:], mode='test', name='')
            image_feat_emb = tf.matmul(image_feat, image_feat_W) + image_feat_b  # B,H
            lstm_input = image_feat_emb
            with tf.variable_scope("lstm_encoder") as scope:
                _, state = encoder(lstm_input, state)
                encoder_state = state

            #============================= decoder ===================================================================

            start_token = tf.constant(self.START, tf.int32, [self.batch_size])
            mask = tf.constant(True, "bool", [self.batch_size])

            log_probs_action_picked_list = []
            sample_mask_list = []
            
            ## modified to run story model
            state = encoder_state
            for j in range(self.lstm_steps):
                with tf.device("/cpu:0"):
                    if j == 0:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, start_token)
                    else:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, self.sample_words_for_loss[:,j-1])

                with tf.variable_scope("lstm"):
                    if not j == 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = decoder(decoder_input, state)

                logits = tf.matmul(output, output_W) + output_b
                log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))  # B,Vocab_size # add 1e-8 to prevent log(0)

                # check if the end of the sentence
                # mask_step -> "current" step, predict has "1 delay"
                sample_mask_list.append(tf.to_float(mask))
                action_picked = tf.range(self.batch_size) * self.vocab_size + tf.to_int32(self.sample_words_for_loss[:,j]) # B
                log_probs_action_picked = tf.multiply(tf.gather(tf.reshape(log_probs, [-1]), action_picked), tf.to_float(mask))  # propabilities for picked actions
                log_probs_action_picked_list.append(log_probs_action_picked)
                prev_mask = mask
                mask_judge = tf.not_equal(self.sample_words_for_loss[:,j], self.END)  # B  # return the bool value 
                mask = tf.logical_and(prev_mask, mask_judge)

            sample_mask_list = tf.stack(sample_mask_list)  # S,B
            sample_mask_list = tf.transpose(sample_mask_list, [1,0])  # B,S
            log_probs_action_picked_list = tf.stack(log_probs_action_picked_list)  # S,B
            log_probs_action_picked_list = tf.reshape(log_probs_action_picked_list, [-1])  # S*B

            loss = -1 * tf.reduce_sum(log_probs_action_picked_list * self.discounted_reward)/ \
                           tf.reduce_sum(sample_mask_list)

            return loss

    def generator(self, name="generator", reuse=False):
        '''
           Caption sampler, sample words follow the probability distribution.

        '''
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.device("/cpu:0"), tf.variable_scope("word"):
                # name: "gnerator/word"
                word_emb_W = tf.get_variable("word_emb_W", [self.vocab_size, self.G_hidden_size], tf.float32, random_uniform_init)
            
            with tf.variable_scope("image_feat"):
                # name: "generator/image_feat"
                image_feat_W = tf.get_variable("image_feat_W", [self.image_feat_dim, self.G_hidden_size], tf.float32, random_uniform_init)
                image_feat_b = tf.get_variable("image_feat_b", [self.G_hidden_size], tf.float32, random_uniform_init)

            with tf.variable_scope("output"):
                # name: "generator/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.vocab_size], tf.float32, random_uniform_init)
                output_b = tf.get_variable("output_b", [self.vocab_size], tf.float32, random_uniform_init)

            with tf.variable_scope("lstm_encoder"):
                if self.rnn_cell == 'lstm':
                    encoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    encoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)

            with tf.variable_scope("lstm_decoder"):
                # WONT BE CREATED HERE
                if self.rnn_cell == 'lstm':
                    decoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    decoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)

            #============================= encoder ===================================================================
            state = encoder.zero_state(self.batch_size, tf.float32)
            with tf.variable_scope("image_feat") as scope:
                image_feat = self.batch_norm(self.image_feat[:,:], mode='train', name='')
            image_feat_emb = tf.matmul(image_feat, image_feat_W) + image_feat_b  # B,H
            lstm_input = image_feat_emb
            with tf.variable_scope("lstm_encoder") as scope:
                _, state = encoder(lstm_input, state)
                encoder_state = state

            #============================= decoder ===================================================================

            start_token = tf.constant(self.START, tf.int32, [self.batch_size])
            mask = tf.constant(True, "bool", [self.batch_size])

            sample_words = []
            
            state = encoder_state
            for j in range(self.lstm_steps):
                with tf.device("/cpu:0"):
                    if j == 0:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, start_token)
                    else:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, sample_word)
                
                with tf.variable_scope("lstm"):
                    if not j == 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = decoder(decoder_input, state)

                    logits = tf.matmul(output, output_W) + output_b
                    log_probs = tf.log(tf.clip_by_value(tf.nn.softmax(logits), 1e-20, 1.0))  # B,Vocab_size # add 1e-8 to prevent log(0)
                    
                    # sample once from the multinomial distribution
                    # Montecarlo sampling
                    sample_word = tf.reshape(tf.multinomial(log_probs, 1), [self.batch_size])   # 1 means sample once
                    sample_words.append(sample_word)

            return sample_words

    def generator_test(self, name="generator", reuse=True):
        '''
           Caption generator. Generate words with max probabilities.

        '''
        random_uniform_init = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
        with tf.variable_scope(name):

            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.device("/cpu:0"), tf.variable_scope("word"):
                # name: "gnerator/word"
                word_emb_W = tf.get_variable("word_emb_W", [self.vocab_size, self.G_hidden_size], tf.float32, random_uniform_init)
            
            with tf.variable_scope("image_feat"):
                # name: "generator/image_feat"
                image_feat_W = tf.get_variable("image_feat_W", [self.image_feat_dim, self.G_hidden_size], tf.float32, random_uniform_init)
                image_feat_b = tf.get_variable("image_feat_b", [self.G_hidden_size], tf.float32, random_uniform_init)

            with tf.variable_scope("output"):
                # name: "generator/output"
                output_W = tf.get_variable("output_W", [self.G_hidden_size, self.vocab_size], tf.float32, random_uniform_init)
                output_b = tf.get_variable("output_b", [self.vocab_size], tf.float32, random_uniform_init)

            with tf.variable_scope("lstm_encoder"):
                if self.rnn_cell == 'lstm':
                    encoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    encoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)

            with tf.variable_scope("lstm_decoder"):
                # WONT BE CREATED HERE
                if self.rnn_cell == 'lstm':
                    decoder = tf.nn.rnn_cell.LSTMCell(self.G_hidden_size, state_is_tuple=True)
                elif self.rnn_cell == 'gru':
                    decoder = tf.nn.rnn_cell.GRUCell(self.G_hidden_size)

            #============================= encoder ===================================================================
            state = encoder.zero_state(self.batch_size, tf.float32)
            with tf.variable_scope("image_feat") as scope:
                image_feat = self.batch_norm(self.image_feat[:,:], mode='test', name='')
            image_feat_emb = tf.matmul(image_feat, image_feat_W) + image_feat_b  # B,H
            lstm_input = image_feat_emb
            with tf.variable_scope("lstm_encoder") as scope:
                _, state = encoder(lstm_input, state)
                encoder_state = state

            #============================= decoder ===================================================================

            start_token = tf.constant(self.START, tf.int32, [self.batch_size])

            sample_words = []
            mask = tf.constant(True, "bool", [self.batch_size])
            
            state = encoder_state
            for j in range(self.lstm_steps):
                with tf.device("/cpu:0"):
                    if j == 0:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, start_token)
                    else:
                        decoder_input = tf.nn.embedding_lookup(word_emb_W, sample_word)

                with tf.variable_scope("lstm"):
                    if not j == 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = decoder(decoder_input, state)

                logits = tf.matmul(output, output_W) + output_b
                probs = tf.nn.softmax(logits)
                log_probs = tf.log(probs + 1e-8)  # B,Vocab_size # add 1e-8 to prevent log(0)
                
                # sample the word with highest probability
                # remove <UNK>
                left = tf.slice(probs, [0,0], [-1,self.UNK]) 
                right = tf.slice(probs, [0,self.UNK+1], [-1,self.vocab_size-self.UNK-1])
                zeros = tf.zeros([self.batch_size, 1])+1e-20
                probs_no_unk = tf.concat([left,zeros,right], 1)
                sample_word = tf.reshape(tf.argmax(probs_no_unk, 1), [self.batch_size])
                sample_words.append(sample_word)

            return sample_words


    def load(self, checkpoint_dir):
        print(' [*] Reading checkpoints ...')
        model_dir = '%s' % self.dataset_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
    
    def decode(self, word_lists, type='string', wo_start=False, rm_end=False):
        '''
        Join the discrete words in word lists to compose sentences.
        Process batch_size*max_words generated stories.
        '''
        if type == 'string':

            processed_sentences = []
            for word_list in word_lists:
                sentence = []
                for word_ix in word_list[0:self.max_words]:
                    sentence.append(self.ix2word[word_ix])
                
                if '</S>' in sentence:
                    punctuation = np.argmax(np.array(sentence) == '</S>') + 1
                    sentence = sentence[:punctuation]
                
                sentence = ' '.join(sentence)
                sentence = sentence.replace('<S> ', '')
                sentence = sentence.replace(' </S>', '')
                sentence = sentence.replace(' \n ', '\n')               
                if sentence.strip() == '':
                    continue
                processed_sentences.append(sentence)

            return processed_sentences

    def load_params(self, ckpt_file):
        tf.global_variables_initializer().run()

        logging.info(Fore.GREEN + 'Init model from %s ...' % ckpt_file)
        G_saver = tf.train.Saver(self.load_G_params_dict)
        G_saver.restore(self.sess, ckpt_file)

    def test_one_image(self, image_feature):
        feed_dict = {self.image_feat: image_feature}
        generated_words = self.sess.run(self.sample_words_argmax, feed_dict)
        generated_sentences = self.decode(generated_words, type='string', wo_start=True, rm_end=True)
        return generated_sentences
