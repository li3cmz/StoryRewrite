from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.tf.module_base import ModuleBase
from texar.tf.core import layers
from tensorflow.python.keras.initializers import glorot_uniform, Zeros
from tensorflow.python.keras.layers import Dense, Layer
from tensorflow.python.keras.regularizers import l2

__all__ = [
    "PoolingAggregator"
]


class PoolingAggregator(ModuleBase): 
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)
        self.dense_layers = [Dense(
            self._hparams.input_dim, activation='relu', use_bias=True, kernel_regularizer=l2(self._hparams.l2_reg))]

        if self._hparams.l2_reg == 0.0:    
            self.regularizer = None
        else:           	
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=self._hparams.l2_reg)
        
        self.neigh_weights = tf.get_variable('neigh_weights',  
                                            [self._hparams.input_dim * 2, self._hparams.output_dim], 
                                            initializer=tf.contrib.layers.xavier_initializer(), 
                                            regularizer=self.regularizer,
                                            trainable=True)                                           ###check?

        if self._hparams.use_bias:
            self.bias = tf.get_variable('bias_weight',  [1, self._hparams.output_dim],	
                            initializer=tf.constant_initializer(0.0),	
                            regularizer=self.regularizer,
                            trainable=True)

    @staticmethod
    def default_hparams():
        return {
            "output_dim": 768,
            "input_dim":768,
            "concat": True,
            "pooling":'meanpooling',
            "dropout_rate":0.0,
            "l2_reg":0.1,
            "use_bias":False,
            "activation":tf.nn.relu,
            "seed":1024,
            "update_weights":False,
            "name": "pooling_aggregator",
            }

    def _build(self, features, self_node_id, neigh_nodes_id):
        """
        Use:
            features: [batch_maxlen-1, dim], self_node_id: int, neigh_nodes_id: [None]
        Create:
            features: [batch_maxlen-1, dim]
        """
        node_feat = tf.nn.embedding_lookup(features, tf.expand_dims(self_node_id, 0))      #[1,dim]
        neigh_feat = tf.nn.embedding_lookup(features, tf.expand_dims(neigh_nodes_id, 0))   #[1,None,dim]

        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(
            neigh_feat, (batch_size * num_neighbors, self._hparams.input_dim))             #[None, dim]

        for l in self.dense_layers:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(
            h_reshaped, (batch_size, num_neighbors, int(h_reshaped.shape[-1])))            #[1, None, dim]

        if self._hparams.pooling == "meanpooling":
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1, keep_dims=False)               #[1,dim]
        elif self._hparams.pooling == "maxpooling":
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        output = tf.concat(
            [node_feat, neigh_feat], axis=-1)                                              #[1,2dim]

        output = tf.matmul(output, self.neigh_weights)                                     #[1,2dim]->[1,dim]
        if self._hparams.use_bias:
            output += self._hparams.use_bias
        if self._hparams.activation:
            output = self._hparams.activation(output)

        # output = tf.nn.l2_normalize(output, dim=-1)
        features1 = tf.concat([features[:tf.cast(self_node_id, tf.int32)], output], 0)
        features2 = features[(tf.cast(self_node_id, tf.int32)+1):]
        features = tf.concat([features1, features2],0)                                   #[batch_maxlen-1, dim]
       
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True
            
        return features