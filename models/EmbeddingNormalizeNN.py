from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib.layers.python.layers import batch_norm
from texar.tf.module_base import ModuleBase
from texar.tf.core import layers

__all__ = [
    "EmbeddingNormalizeNN"
]


class EmbeddingNormalizeNN(ModuleBase): 
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        with tf.variable_scope(self._hparams.name_scope):
            self.epsilon = self._hparams.epsilon
            self.decay = self._hparams.decay
            size = self._hparams.size

            self.scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(0.1), trainable=True)
            self.offset = tf.get_variable('offset', [size], trainable=True)
            self.pop_mean = tf.get_variable('pop_mean', [size], initializer=tf.zeros_initializer(), trainable=False)
            self.pop_var = tf.get_variable('pop_var', [size], initializer=tf.ones_initializer(), trainable=False)
            
        

    @staticmethod
    def default_hparams():
        return {
            "name": "EmbeddingNormalizeNN",
            "max_seqlen":16,
            "size":512,
            "epsilon": 1e-3,
            "decay":0.99,
            "name_scope":'nnBN'
            }

    def _build(self, features, training):
        """Use:Compute the similarity between two nodes

        Args:
            inputs: 
                node_feat: [batch_size, maxlen-1, dim]
        Returns:
            node_feat:[batch_size, maxlen-1, dim]
            
        """
        """ Assume nd [batch, N1, N2, ..., Nm, Channel] tensor"""
        self.x = features
        self.batch_mean, self.batch_var = tf.nn.moments(self.x, list(range(len(self.x.get_shape())-1)))
        self.train_mean_op = tf.assign(self.pop_mean, self.pop_mean * self.decay + self.batch_mean * (1 - self.decay))
        self.train_var_op = tf.assign(self.pop_var, self.pop_var * self.decay + self.batch_var * (1 - self.decay))
        
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return tf.cond(training, self.batch_statistics, self.population_statistics)

    def batch_statistics(self):
        with tf.control_dependencies([self.train_mean_op, self.train_var_op]):
            return tf.nn.batch_normalization(self.x, self.batch_mean, self.batch_var, self.offset, self.scale, self.epsilon)
    def population_statistics(self):
        return tf.nn.batch_normalization(self.x, self.pop_mean, self.pop_var, self.offset, self.scale, self.epsilon)