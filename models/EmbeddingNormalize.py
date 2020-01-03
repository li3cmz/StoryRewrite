from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib.layers.python.layers import batch_norm
from texar.module_base import ModuleBase
from texar.core import layers

__all__ = [
    "EmbeddingNormalize"
]


class EmbeddingNormalize(ModuleBase): 
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)
        #self.bn1 = tf.keras.layers.BatchNormalization()
    @staticmethod
    def default_hparams():
        return {
            "name": "EmbeddingNormalize",
            "max_seqlen":16
            }

    def _build(self, features, training):
        """Use:Compute the similarity between two nodes

        Args:
            inputs: 
                node_feat: [batch_size, maxlen-1, dim]
        Returns:
            node_feat:[batch_size, maxlen-1, dim]
            
        """
        #bn1 = tf.layers.batch_normalization(features, training=training)
        '''
        bn1 = self.bn1(features, training=training)
        self.op_num = len(self.bn1.updates)#2
        placeholder = 3
        self.op_num = tf.while_loop(self.cond, self.body, loop_vars=[placeholder])
        '''
        #relu1 = tf.nn.leaky_relu(bn1)
        bn1 = tf.contrib.layers.batch_norm(features, is_training=training, updates_collections=None)
        
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return  bn1
    '''
    def cond(self, placeholder):
        return tf.greater(self.op_num, 0)
    def body(self, placeholder):
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.bn1.updates[self.op_num-1])
        self.op_num-=1

        return placeholder
    '''