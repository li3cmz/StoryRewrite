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
    "PRelu"
]


class PRelu(ModuleBase): 
    def __init__(self, hparams=None):
        ModuleBase.__init__(self, hparams)

        self.alphas = tf.get_variable('alpha', self._hparams.dim, initializer=tf.constant_initializer(0.0), trainable=True,
                                 dtype=tf.float32)

    @staticmethod
    def default_hparams():
        return {
            "dim": 512,
            "name": "PRelu",
            }

    def _build(self, _x):

        pos = tf.nn.relu(_x)
        neg = self.alphas * (_x - tf.abs(_x)) * 0.5


        if not self._built:
            self._add_internal_trainable_variables()
            self._built = True

        return pos + neg
    
        
        
