from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.module_base import ModuleBase
from texar.core import layers
from models.inits import *

__all__ = [
    "GCNAggregator"
]


class GCNAggregator(ModuleBase): 
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        self.dropout = self._hparams.dropout
        self.bias = self._hparams.bias
        self.act = self._hparams.act
        self.concat = self._hparams.concat
        self.vars = {}
        #self.logging = True

        if self._hparams.neigh_input_dim is None:
            neigh_input_dim = self._hparams.input_dim

        if self._hparams.name is not None:
            name = '/' + self._hparams.name
        else:
            name = ''

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neigh_input_dim, self._hparams.output_dim],
                                                        name='neigh_weights')
            if self.bias:
                self.vars['bias'] = zeros([self._hparams.output_dim], name='bias')

        '''
        if self.logging:
            self._log_vars()
        '''

        self.input_dim = self._hparams.input_dim
        self.output_dim = self._hparams.output_dim

    @staticmethod
    def default_hparams():
        return {
            "input_dim": 512,
            "output_dim": 512,
            "neigh_input_dim": None,
            "dropout":0.,
            "bias":False, 
            "act":tf.nn.relu,
            "name":None, 
            "concat":False, 
            "name": "GCNAggregator"
            }

    def _build(self, node_vecs, neigh_vecs):
        #node_vecs, neigh_vecs = inputs # [b*max, inputdim] [b*max, max, dim_edge] -> 
        
        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        node_vecs = tf.nn.dropout(node_vecs, 1-self.dropout)
        '''
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
            tf.expand_dims(node_vecs, axis=1)], axis=1), axis=1) # [b*max, 1, dim] [b*max, max, dim] # only consider out degree
                                                                # [b*max, 1+max, dim]->[b*max, dim]
        '''
        relation_means = tf.reduce_mean(neigh_vecs, 1) # [b*max, dim]
        new_node_vecs = tf.multiply(node_vecs, relation_means) # [b*max, dim]
       
        # [nodes] x [out_dim]
        output = tf.matmul(new_node_vecs, self.vars['weights']) # [b*max, out_dim] # padding is still 0

        # bias
        if self.bias:
            output += self.vars['bias'] # [b*max, out_dim]

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return self.act(output) # [b*max, out_dim]