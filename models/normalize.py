from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from tensorflow.contrib.layers.python.layers import batch_norm
from texar.module_base import ModuleBase
from texar.core import layers

__all__ = [
    "Normalize"
]


class Normalize(ModuleBase): 
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

    @staticmethod
    def default_hparams():
        return {
            "name": "Normalize",
            "max_seqlen":16
            }

    def _build(self, features):
        """Use:Compute the similarity between two nodes

        Args:
            inputs: 
                node_feat: [batch_size, max_seq_len, max_seq_len]
        Returns:
            sim_val:[batch_size, max_seq_len, max_seq_len]
            
        """
        tf_shape = tf.shape(features)
        features = tf.reshape(features, [-1,1])
        #bn1 = batch_norm(features,decay=0.9,updates_collections=None,is_training=True)
        bn1 = tf.layers.batch_normalization(features)
        #relu1 = tf.nn.leaky_relu(bn1)
        bn1 = tf.reshape(bn1, [tf_shape[0], tf_shape[1], tf_shape[1]])
        
        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return  bn1