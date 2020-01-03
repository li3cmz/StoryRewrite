from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from texar.module_base import ModuleBase
from texar.core import layers

__all__ = [
    "Transform"
]


class Transform(ModuleBase): ###check CNN or RNN ###check
    """
    For Compute the similarity between two node
    """
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)


    @staticmethod
    def default_hparams():
        return {
            "name": "Transform",
            "input_dim":1024,
            "output_dim":512
            }

    def _build(self, inputs):
        """Use:Compute the similarity between two nodes

        Args:
            inputs: 
                pre_embedding_text_ids_mat:[batch, maxlen, 2dim]
        Returns:
            text_ids_embedding:[batch, maxlen, dim]
            
        """
        shape = tf.shape(inputs)
        w_trans = tf.get_variable("w_trans", [self._hparams.input_dim, self._hparams.output_dim], dtype=tf.float32)
        b_trans = tf.get_variable("b_trans", [self._hparams.output_dim], dtype=tf.float32)

        pre_embedding_text_ids_mat = tf.reshape(inputs,
                    [-1, self._hparams.input_dim])
        pre_embedding_text_ids_mat = tf.matmul(pre_embedding_text_ids_mat, w_trans) + b_trans
        pre_embedding_text_ids_mat = tf.tanh(pre_embedding_text_ids_mat)

        text_ids_embedding = tf.reshape(pre_embedding_text_ids_mat, [shape[0], shape[1], self._hparams.output_dim])

        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return  text_ids_embedding