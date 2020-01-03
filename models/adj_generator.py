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
"""Text style transfer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import tensorflow as tf

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier, BidirectionalRNNEncoder
from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix
=======
>>>>>>> d21bd78bcb0d566a750d5a920af7a6adeac79b80


class RELA_CLASS(object):
    """Control
    """

<<<<<<< HEAD
    def __init__(self, inputs, vocab, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._prepare_inputs(inputs, vocab)
        self._build_model()

    def _prepare_inputs(self, inputs, vocab):
        self.vocab = vocab
=======
    def __init__(self, inputs, vocab, gamma, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._prepare_inputs(inputs, vocab, gamma)
        self._build_model()

    def _prepare_inputs(self, inputs, vocab, gamma):
        self.vocab = vocab
        self.gamma = gamma
>>>>>>> d21bd78bcb0d566a750d5a920af7a6adeac79b80

        # the first token is the BOS token
        self.text_ids = inputs['text_ids']
        self.sequence_length = inputs['length']
        self.labels = inputs['labels']

        enc_shape = tf.shape(self.text_ids)
        adjs = tf.to_int32(tf.reshape(inputs['adjs'], [-1,17,17]))
        self.adjs = adjs[:, :enc_shape[1], :enc_shape[1]]

    def _build_model(self):
        """Builds the model.
        """
        self._prepare_modules()
        self._build_self_encoder()
        self._get_loss_train_op()

    def _prepare_modules(self):
        """Prepare necessary modules
        """
        self.embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            hparams=self._hparams.embedder)
        self.encoder = BidirectionalRNNEncoder(hparams=self._hparams.encoder)

    def _build_self_encoder(self):
<<<<<<< HEAD
        """Preds adjs matrix
        Use:
            self.embedder, self.encoder
            self.text_ids
        Create:
            self.pred_adjs
        """
        sentence_embedding = self.embedder(self.text_ids)[:, :, :] # [batch_size, max_seq_len, 512]
        outputs, _ = self.encoder(sentence_embedding) # [batch_size, max_seq_len, 512], # [batch_size, max_seq_len, 512])
        sentence_hidden = tf.concat([outputs[0], outputs[1]],2) # [batch_size, max_seq_len, 1024]
        f1 = tf.layers.conv1d(sentence_hidden, 128, kernel_size=3, strides=1, padding='same') # it should be turn to [batch_size, max_seq_len, 1024]
        f2 = tf.layers.conv1d(f1, 256, kernel_size=3, strides=1, padding='same') # it should be turn to [batch_size, max_seq_len, 1024]
        f2_norm = tf.layers.batch_normalization(f2, training=False)
        f3 = tf.layers.conv1d(f2_norm, 512, kernel_size=3, strides=1, padding='same') # it should be turn to [batch_size, max_seq_len, 1024]
        f3_norm = tf.layers.batch_normalization(f3, training=False)
        f4 = tf.layers.conv1d(f3_norm, 512, kernel_size=3, strides=1, padding='same') # it should be turn to [batch_size, max_seq_len, 1024]
        f4_norm = tf.layers.batch_normalization(f4, training=False)
        sentence_hidden = tf.layers.conv1d(f4_norm, 1024, kernel_size=3, strides=1, padding='same') # it should be turn to [batch_size, max_seq_len, 1024]

        # predicted adjacency matrices
        self.pred_adjs = tf.matmul(sentence_hidden, tf.transpose(sentence_hidden, perm=[0, 2, 1])) # [batch_size, max_seq_len, max_seq_len]
        #self.pred_adjs = tf.Print(self.pred_adjs, ["self.pred_adjs: ",tf.rint(tf.sparse.softmax(self.pred_adjs))])
        self._train_generator()
        
=======
        """
        Use:
            self.embedder, self.encoder
            self.text_ids, self.sequence_length
        Create:
            self.pre_embedding_text_ids,
            self.embedding_text_ids
            self.batch_enc_outputs
            self.target
            self.batch_seqlen
        """
        sentence_embedding = self.embedder(self.text_ids)[:, :, :] # [batch_size, max_seq_len, 512]
        outputs, _ = self.encoder(sentence_embedding)
        sentence_hidden = tf.concat(outputs[0], outputs[1]) # [batch_size, max_seq_len, 1024]
        # predicted adjacency matrices
        self.pred_adjs = tf.matmul(sentence_hidden, tf.transpose(sentence_hidden, perm=[0, 2, 1])) # [batch_size, max_seq_len, max_seq_len]
        self._train_generator()
>>>>>>> d21bd78bcb0d566a750d5a920af7a6adeac79b80

    def _train_generator(self):
        """MSE loss for adjacency matrix generator
        Use:
            self.pred_adjs: predicted adjacency matrices # [batch_size, max_seq_len, max_seq_len]
            self.adjs:      target adjacency matrices    # [batch_size, max_seq_len, max_seq_len]
        Create:
            self.loss_adj, self.accu_adj
        """
<<<<<<< HEAD
        pred_adjs_sigmoid = tf.math.sigmoid(self.pred_adjs)
        self.loss_adj = tf.losses.mean_squared_error(labels=self.adjs,
                                                         predictions=pred_adjs_sigmoid)

        self.accu_adj = tx.evals.accuracy(
            labels = self.adjs,
            preds =  tf.rint(pred_adjs_sigmoid)# convert logits into 0-1 value
        )
        self.pred_adjs_binary = tf.rint(pred_adjs_sigmoid)

=======
        self.loss_adj = tf.losses.mean_squared_error(labels=self.adjs,
                                                         predictions=self.pred_adjs)
        self.accu_adj = tx.evals.accuracy(
            labels = self.adjs,
            preds = tf.rint(self.pred_adjs) # convert logits into 0-1 value
        )
>>>>>>> d21bd78bcb0d566a750d5a920af7a6adeac79b80

    def _get_loss_train_op(self):
        # Creates optimizers
        d_vars = collect_trainable_variables([self.embedder, self.encoder])

        train_op_adj = get_train_op(
            self.loss_adj, d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.losses = {
            "loss_adj": self.loss_adj
        }
        self.metrics = {
            "accu_adj": self.accu_adj,
        }
        self.train_ops = {
            "train_op_adj": train_op_adj
        }
        '''
        self.samples = {
            "original": inputs['text_ids'][:, :],
            "transferred": outputs_.sample_id
        }
        '''
        self.fetches_train_d = {
            "loss_adj": self.train_ops["train_op_adj"],
<<<<<<< HEAD
            "accu_adj": self.metrics["accu_adj"],
            "adjs_truth": self.adjs,
            "adjs_preds": self.pred_adjs_binary
        }
        fetches_eval = {
            "batch_size": get_batch_size(self.text_ids),
            "adjs_truth": self.adjs,
            "adjs_preds": self.pred_adjs_binary
        }
=======
            "accu_adj": self.metrics["accu_adj"]
        }
        fetches_eval = {"batch_size": get_batch_size(self.text_ids)}
>>>>>>> d21bd78bcb0d566a750d5a920af7a6adeac79b80
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        '''
        fetches_eval.update(self.samples)
        '''
        self.fetches_eval = fetches_eval