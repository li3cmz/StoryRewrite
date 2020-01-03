# Copyright 2018 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text style transfer Under Linguistic Constraints
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import tensorflow as tf

import texar as tx
from texar.core import get_train_op
from texar.modules import WordEmbedder
from texar.modules import TransformerEncoder
from texar.modules import MLPTransformConnector
from texar.modules import BidirectionalRNNEncoder
from texar.modules import UnidirectionalRNNEncoder
from texar.utils import collect_trainable_variables, get_batch_size

from models.utils.model_util import *
from models.PoolingAggregator import PoolingAggregator
from models.EmbeddingNormalizeNN import EmbeddingNormalizeNN

from models.self_graph_transformer import SelfGraphTransformerEncoder
from models.cross_graph_transformer import CrossGraphTransformerFixedLengthDecoder

from models.rnn_dynamic_decoders import DynamicAttentionRNNDecoder
from models.adj_multi_attention import AdjMultiheadAttentionEncoder
from models.transformer_decoderToEncoder import TransformerDecoderToEncoder
from models.prelu import PRelu

class EvolveGTAE(object):
    """Control  
    """
    def __init__(self, inputs, vocab, ctx_maxSeqLen, lr, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._prepare_inputs(inputs, vocab, ctx_maxSeqLen, lr)
        self._build_model()
        
    
    def _prepare_inputs(self, inputs, vocab, ctx_maxSeqLen, lr):
        self.lr=lr
        self.vocab = vocab
        self.ctx_maxSeqLen = ctx_maxSeqLen+2                  #inlcude BOS and EOS

        
        self.x1x2 = inputs['ctx_text_ids'][:,0,:]
        self.x1xx2 = inputs['ctx_text_ids'][:,1,:]
        self.x1x2yx1xx2_ids = inputs['ctx_text_ids'][:,2,:]     # x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2 + ' ' 
        self.x1x2yx1xx2yy_ids = inputs['ctx_text_ids'][:,3,:]   # x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2 + ' ' 
        self.x1x2y_ids = inputs['ctx_text_ids'][:,4,:] 
        self.x1_ids = inputs['ctx_text_ids'][:,5,:]
        self.x1x2yx1my_ids = inputs['ctx_text_ids'][:,6,:]
        self.x1x2yx1m_ids = inputs['ctx_text_ids'][:,7,:]
        
        self.text_ids_y1 = inputs['y1_yy1_text_ids'][:,0,:]    # [batch, maxlen1]
        self.text_ids_yy1 = inputs['y1_yy1_text_ids'][:,1,:]  # [batch, maxlen1]
        self.text_ids_y2 = inputs['y2_yy2_text_ids'][:,0,:]    # [batch, maxlen2]
        self.text_ids_yy2 = inputs['y2_yy2_text_ids'][:,1,:]  # [batch, maxlen2]
        self.text_ids_y3 = inputs['y3_yy3_text_ids'][:,0,:]    # [batch, maxlen3]
        self.text_ids_yy3 = inputs['y3_yy3_text_ids'][:,1,:]  # [batch, maxlen3]

        sequence_length_x = inputs['ctx_length']
        self.x1x2_truth_len = sequence_length_x[:,0]    #[batch]
        self.x1xx2_truth_len = sequence_length_x[:,1] #[batch]
        self.x1x2yx1xx2_truth_len = sequence_length_x[:,2] #[batch]
        self.x1x2yx1xx2yy_truth_len = sequence_length_x[:,3] #[batch]
        self.x1x2y_truth_len = sequence_length_x[:,4] #[batch]
        self.x1_truth_len = sequence_length_x[:,5] #[batch]
        self.x1x2yx1my_truth_len = sequence_length_x[:,6]
        self.x1x2yx1m_truth_len = sequence_length_x[:,7]

        self.sequence_length_y1 = inputs['y1_yy1_length'][:,0]
        self.sequence_length_y2 = inputs['y2_yy2_length'][:,0]
        self.sequence_length_y3 = inputs['y3_yy3_length'][:,0]
        self.sequence_length_yy1 = inputs['y1_yy1_length'][:,1] # pad_to_maxlen
        self.sequence_length_yy2 = inputs['y2_yy2_length'][:,1]
        self.sequence_length_yy3 = inputs['y3_yy3_length'][:,1]
        
        enc_shape_y1 = tf.shape(self.text_ids_y1)[1]
        enc_shape_y2 = tf.shape(self.text_ids_y2)[1]
        enc_shape_y3 = tf.shape(self.text_ids_y3)[1]
        
        self.adjs_y1_dirt = tf.to_int32(tf.reshape(inputs['y1_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y1, :enc_shape_y1]
        self.adjs_y2_dirt = tf.to_int32(tf.reshape(inputs['y2_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y2, :enc_shape_y2]
        self.adjs_y3_dirt = tf.to_int32(tf.reshape(inputs['y3_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y3, :enc_shape_y3]
        self.adjs_yy1_dirt = tf.to_int32(tf.reshape(inputs['yy1_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y1, :enc_shape_y1]
        self.adjs_yy2_dirt = tf.to_int32(tf.reshape(inputs['yy2_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y2, :enc_shape_y2]
        self.adjs_yy3_dirt = tf.to_int32(tf.reshape(inputs['yy3_d_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y3, :enc_shape_y3]
       
        self.adjs_y1_undirt = tf.to_int32(tf.reshape(inputs['y1_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y1, :enc_shape_y1]
        self.adjs_y2_undirt = tf.to_int32(tf.reshape(inputs['y2_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y2, :enc_shape_y2]
        self.adjs_y3_undirt = tf.to_int32(tf.reshape(inputs['y3_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y3, :enc_shape_y3]
        self.adjs_yy1_undirt = tf.to_int32(tf.reshape(inputs['yy1_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y1, :enc_shape_y1]
        self.adjs_yy2_undirt = tf.to_int32(tf.reshape(inputs['yy2_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y2, :enc_shape_y2]
        self.adjs_yy3_undirt = tf.to_int32(tf.reshape(inputs['yy3_und_adjs'], [-1,self.ctx_maxSeqLen,self.ctx_maxSeqLen]))[:, :enc_shape_y3, :enc_shape_y3]

        self.gamma = 0.5

    def _build_model(self):
        """Builds the model.
        """
        self._prepare_modules()
        self._build_ctx_encoder()
        self.loss_fine = self._build_auxiliary_loss(self.x1x2y_ids, self.x1x2y_truth_len, self.x1x2_truth_len, prefix=False)
        self.loss_xx2 = self._build_auxiliary_loss(self.x1xx2, self.x1xx2_truth_len, self.x1_truth_len)
        self.loss_mask_recon = self._build_auxiliary_loss(self.x1x2yx1my_ids, self.x1x2yx1my_truth_len, self.x1x2yx1m_truth_len)

        self._overallPipeline(self.text_ids_y1, self.text_ids_yy1, self.sequence_length_y1, self.sequence_length_yy1, self.adjs_y1_undirt, self.adjs_yy1_undirt)
        self._overallPipeline(self.text_ids_y2, self.text_ids_yy2, self.sequence_length_y2, self.sequence_length_yy2, self.adjs_y2_undirt, self.adjs_yy2_undirt)
        self._overallPipeline(self.text_ids_y3, self.text_ids_yy3, self.sequence_length_y3, self.sequence_length_yy3, self.adjs_y3_undirt, self.adjs_yy3_undirt)
        self._get_loss_train_op()
    

    def _overallPipeline(self, y_text_ids, yy_text_ids, y_truth_len, yy_truth_len, adjs_y_undirt, adjs_yy_undirt):
        self._build_origin_graph_y_encoder(y_text_ids, y_truth_len, adjs_y_undirt)
        self._build_cft_graph_y_encoder(y_truth_len, yy_text_ids, yy_truth_len)


    def _prepare_modules(self):
        """Prepare necessary modules
        """
        self.training = tx.context.global_mode_train()                                                                 ## 判断当前是否在训练
        
        # encode ctx
        self.transformer_encoder = TransformerEncoder(hparams=self._hparams.transformer_encoder)
        
        # encode y
        self.word_embedder = WordEmbedder(
            vocab_size = self.vocab.size, 
            hparams=self._hparams.wordEmbedder
        )
        self.self_graph_encoder = SelfGraphTransformerEncoder(hparams=self._hparams.encoder)

        self.downmlp = MLPTransformConnector(self._hparams.dim_c)
        self.PRelu = PRelu(self._hparams.prelu)

        self.rephrase_encoder = UnidirectionalRNNEncoder(hparams=self._hparams.rephrase_encoder)
        self.rephrase_decoder = DynamicAttentionRNNDecoder(
            memory_sequence_length = self.sequence_length_yy1-1,                                                       ## use yy1's truth length ###check?
            cell_input_fn = lambda inputs, attention: inputs,
            vocab_size = self.vocab.size,
            hparams = self._hparams.rephrase_decoder
        )


    def _build_origin_graph_y_encoder(self, y_text_ids, y_truth_len, adjs_y_undirt):
        """
        Use: 
            y_text_ids(y1,y2,y3), y_truth_len
        Create:
            self.y_embd [batch, maxlen, dim], self.graph_y_embd [batch, maxlen-1, dim]
        """
        self.y_embd = self.self_graph_encoder(
            inputs = self.word_embedder(y_text_ids)[:,1:,:], 
            sequence_length = y_truth_len-1, 
            adjs = adjs_y_undirt[:,1:,1:]
        )

    def _build_auxiliary_loss(self, input_ids, full_len, prefix_len, prefix=True):

        input_embd = self.transformer_encoder(
            inputs= self.word_embedder(input_ids),
            sequence_length = full_len
        )[:,1:,:]
        rephrase_enc, rephrase_state = self.rephrase_encoder(
            input_embd                                                                                     # [batch, maxlen-1, :] 
        )
        outputs, _, _ = self.rephrase_decoder(                                           # decode for training loss
            initial_state = rephrase_state,
            memory = rephrase_enc,
            sequence_length = full_len-1,
            inputs = input_ids,                                                                                         # [batch, maxlen-1]
            embedding = self.word_embedder
        )

        max_full_len = tf.reduce_max(full_len)
        ids = input_ids#[:, :max_full_len]
        logits = outputs.logits#[:, :max_full_len]
        if prefix==False:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:(tf.reduce_max(full_len))],
                logits=logits,
                sequence_length=full_len-1,
                average_across_timesteps=True,
                sum_over_timesteps=False,
                average_across_batch=True,
                sum_over_batch=False
            )    
        else:
            loss_recon = tx.losses.sequence_sparse_softmax_cross_entropy(
                labels=ids[:, 1:(tf.reduce_max(full_len))],
                logits=logits,
                sequence_length=full_len-1,
                average_across_timesteps=False,
                sum_over_timesteps=False,
                average_across_batch=False,
                sum_over_batch=False
            )
            mask_recon = tf.sequence_mask(
                full_len-1,
                dtype=tf.float32)
            mask_recon_prefix = 1 - tf.sequence_mask(
                prefix_len-1,
                maxlen=max_full_len-1,#max_decoding_length-1,
                dtype=tf.float32)
            mask_recon = mask_recon * mask_recon_prefix
            loss_recon = tx.utils.reduce_with_weights(
                    tensor=loss_recon,
                    weights=mask_recon,
                    average_across_remaining=True,
                    sum_over_remaining=False)
        return loss_recon


    def _build_ctx_encoder(self):
        """Update every turn
        Use:
            self.origin_ctx_ids transfer from [batch, maxlen] self.origin_ctx_truth_len: [batch]
            x1x2 -> y1 -> y2 -> y3
            self.countFact_ctx_ids transfer from [batch, maxlen] self.countFact_ctx_truth_len: [batch]
            x1xx2 -> y1' -> y2' -> y3'
        Create:
            self.hidden_state [batch, 1, dim]
        """
        # ctx
        self.ctx_embd = self.transformer_encoder(
            inputs= self.word_embedder(self.x1x2yx1xx2_ids),
            sequence_length = self.x1x2yx1xx2_truth_len
        )                                                                                                           # [batch, maxlen, dim]                                                                           

        
    

    def _build_cft_graph_y_encoder(self, y_truth_len, yy_text_ids, yy_truth_len):
        """
        Use: self.y_embd[:,1:,:]: [batch, maxlen-1, dim], self.hidden_state:[batch, 1, dim]
        Create: self.graph_y_ctx_embd_finally [batch, maxlen, dim]
                self.origin_ctx_ids, self.origin_ctx_truth_len
        """         
        yy_pred_output = self.downmlp(
            tf.reshape(tf.concat([self.ctx_embd[:,1:,:], self.y_embd], 2), [-1, self._hparams.dim_c*2])
        )
        self.yy_pred_output = self.PRelu(tf.reshape(yy_pred_output, [tf.shape(self.y_embd)[0], -1, self._hparams.dim_c]))
        
        self._train_rephraser(yy_text_ids, yy_truth_len)


    def _train_rephraser(self, yy_text_ids, yy_truth_len):
        """Classification loss for the tansferred generator
        Use: 
            self.graph_y_ctx_embd_finally: [batch, maxlen+1, dim] 
            self.max_sequence_length_finally: [batch], self.yy_ids: [batch, maxlen]
        Create:
            self.countFact_ctx_ids, self.countFact_ctx_truth_len
        """
        start_tokens =  tf.ones_like((self.x1x2yx1xx2_truth_len)) * self.vocab.bos_token_id                               #[batch]
        end_token = self.vocab.eos_token_id 
        

        rephrase_enc, rephrase_state = self.rephrase_encoder(
            self.yy_pred_output                                                                                     # [batch, maxlen-1, :] 
        )
        self.rephrase_outputs, _, self.rephrase_length = self.rephrase_decoder(                                           # decode for training loss
            initial_state = rephrase_state,
            memory = rephrase_enc,
            sequence_length = yy_truth_len-1,
            inputs = yy_text_ids,                                                                                         # [batch, maxlen-1]
            embedding = self.word_embedder
        )
        self.loss_rephraser = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels = yy_text_ids[:, 1:(tf.reduce_max(yy_truth_len))],                                                     # [batch, maxlen-1]
            logits = self.rephrase_outputs.logits,                                                                        # [batch,maxlen-1]
            sequence_length = yy_truth_len-1,
            average_across_timesteps = True,
            sum_over_timesteps = False
        )
        tf.add_to_collection('loss_rephraser_list', self.loss_rephraser)


        # for infer
        self.rephrase_outputs_, _, self.rephrase_length_ = self.rephrase_decoder(                                         # decode for eval and test
            decoding_strategy = 'infer_greedy',
            memory = rephrase_enc,
            initial_state = rephrase_state,
            embedding = self.word_embedder,                                                                         
            start_tokens = start_tokens, ###debug
            end_token = end_token
        )
        
        tf.add_to_collection('yy_gt_list', yy_text_ids)
        tf.add_to_collection('yy_pred_list', self.rephrase_outputs_.sample_id)
        
        

    def _get_loss_train_op(self):
        # Aggregates loss
        loss_rephraser =  (tf.get_collection('loss_rephraser_list')[0] + tf.get_collection('loss_rephraser_list')[1] + tf.get_collection('loss_rephraser_list')[2])/3.
        w_recon = 1.0
        w_fine = 0.5
        w_xx2 = 0.0
        self.loss = loss_rephraser + w_fine*self.loss_fine + w_recon*self.loss_mask_recon #+ w_xx2*self.loss_xx2###check逐量级修改
        
        # Creates optimizers
        self.vars = collect_trainable_variables([self.transformer_encoder, self.word_embedder, self.self_graph_encoder,
                self.downmlp, self.PRelu, self.rephrase_encoder, self.rephrase_decoder])
                
        # Train Op
        self.train_op_pre = get_train_op(
                self.loss, self.vars, hparams=self._hparams.opt)#learning_rate=self.lr
        
        # Interface tensors
        self.losses = {
            "loss": self.loss,
            "loss_rephraser": loss_rephraser,
            "loss_fine":self.loss_fine,
            "loss_mask_recon":self.loss_mask_recon,
            "loss_xx2": self.loss_xx2,
        }
        self.metrics = {
        }
        self.train_ops = {
            "train_op_pre": self.train_op_pre,
        }
        self.samples = {
            "transferred_yy1_gt": self.text_ids_yy1,
            "transferred_yy1_pred": tf.get_collection('yy_pred_list')[0],
            "transferred_yy2_gt": self.text_ids_yy2,
            "transferred_yy2_pred": tf.get_collection('yy_pred_list')[1],
            "transferred_yy3_gt": self.text_ids_yy3,
            "transferred_yy3_pred": tf.get_collection('yy_pred_list')[2],
            "origin_y1":self.text_ids_y1,
            "origin_y2":self.text_ids_y2,
            "origin_y3":self.text_ids_y3,
            "x1x2":self.x1x2,
            "x1xx2":self.x1xx2
        }

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("loss_rephraser", loss_rephraser)
        tf.summary.scalar("loss_fine", self.loss_fine)
        tf.summary.scalar("loss_mask_recon", self.loss_mask_recon)
        tf.summary.scalar("loss_xx2", self.loss_xx2)
        self.merged = tf.summary.merge_all()
        self.fetches_train_pre = {
            "loss": self.train_ops["train_op_pre"],
            "loss_rephraser": self.losses["loss_rephraser"],
            "loss_fine": self.losses["loss_fine"],
            "loss_mask_recon": self.losses["loss_mask_recon"],
            "loss_xx2": self.losses["loss_xx2"],
            "merged": self.merged,
        }
        fetches_eval = {"batch_size": get_batch_size(self.x1x2yx1xx2_ids),
        "merged": self.merged,
        }
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        self.fetches_eval = fetches_eval