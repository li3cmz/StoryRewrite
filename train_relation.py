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
This is a simplified implementation of:
Toward Controlled Generation of Text, ICML2017
Zhiting Hu, Zichao Yang, Xiaodan Liang, Ruslan Salakhutdinov, Eric Xing
Download the data with the cmd:
$ python prepare_data.py
Train the model with the cmd:
$ python main.py --config config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import sys
import importlib
import numpy as np
import tensorflow as tf
import texar as tx
import argparse
import importlib
import pdb
import logging


from models.adj_generator import RELA_CLASS
from utils_data.multi_aligned_data_with_numpy import MultiAlignedNumpyData
from utils_data.utils import *
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

flags = tf.flags

flags.DEFINE_string('config', 'config_for_clas', 'The config to use.')
flags.DEFINE_string('out', 'tmp1', 'The file to save output.')
FLAGS = flags.FLAGS

config = importlib.import_module(FLAGS.config)
output_path = FLAGS.out
if output_path == 'none':
    raise ValueError('output path is not specified. E.g. python main.py --out output_path')


# get logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger_format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logger_format = logging.Formatter(logger_format_str)
logger_sh = logging.StreamHandler()
logger_sh.setFormatter(logger_format)
logger_th = logging.FileHandler('{}.log'.format(output_path), mode='w')
logger_th.setFormatter(logger_format)
logger.addHandler(logger_sh)
logger.addHandler(logger_th)

logger.info('config: {}.py'.format(FLAGS.config))

def _main(_):

    '''
    # Create output_path
    if os.path.exists(output_path):
        logger.error('output path {} already exists'.format(output_path))
        raise ValueError('output path {} already exists'.format(output_path))
    os.mkdir(output_path)
    os.mkdir('{}/src'.format(output_path))
    os.system('cp *.py {}/src'.format(output_path))
    os.system('cp models/*.py {}/src'.format(output_path))
    os.system('cp utils_data/*.py {}/src'.format(output_path))
    '''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir('{}/src'.format(output_path))
        os.system('cp *.py {}/src'.format(output_path))
        os.system('cp models/*.py {}/src'.format(output_path))
        os.system('cp utils_data/*.py {}/src'.format(output_path))

    # clean sample_path and checkpoint_path before training 
    if tf.gfile.Exists(config.sample_path):
        tf.gfile.DeleteRecursively(config.sample_path)
    if tf.gfile.Exists(config.checkpoint_path):
        tf.gfile.DeleteRecursively(config.checkpoint_path)
    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)
    
    # Data
    train_data = MultiAlignedNumpyData(config.train_data)
    val_data = MultiAlignedNumpyData(config.val_data)
    test_data = MultiAlignedNumpyData(config.test_data)
    vocab = train_data.vocab(0)
 

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.FeedableDataIterator(
        {'train': train_data,
         'val': val_data, 'test': test_data})
    batch = iterator.get_next()

    # Model
    model = RELA_CLASS(batch, vocab, config.model)

    def _train_epoch(sess, epoch, adjs_true_list, adjs_preds_list, verbose=True):
        avg_meters_d = tx.utils.AverageRecorder(size=10)

        step = 0
        while True:
            try:
                step += 1
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train'),
                }
                vals_d = sess.run(model.fetches_train_d, feed_dict=feed_dict)
                adjs_truth = np.reshape(vals_d.pop("adjs_truth"), [-1]) # [128,17,17]
                adjs_preds = np.reshape(vals_d.pop("adjs_preds"), [-1])
                adjs_true_list.extend(adjs_truth)
                adjs_preds_list.extend(adjs_preds)
                avg_meters_d.add(vals_d)


                if verbose and (step == 1 or step % config.display == 0):
                    logger.info('step: {}, {}'.format(step, avg_meters_d.to_str(4)))

                if verbose and step % config.display_eval == 0:
                    iterator.restart_dataset(sess, 'val')
                    tmp_a = []
                    tmp_b = []
                    _eval_epoch(sess, epoch, tmp_a, tmp_b)

            except tf.errors.OutOfRangeError:
                logger.info('epoch: {}, {}'.format(epoch, avg_meters_d.to_str(4)))
                break

        return adjs_true_list, adjs_preds_list

    def _eval_epoch(sess, epoch, adjs_true_list, adjs_preds_list, val_or_test='val'):
        avg_meters = tx.utils.AverageRecorder()

        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, val_or_test),
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }

                vals = sess.run(model.fetches_eval, feed_dict=feed_dict)
                adjs_truth = np.reshape(vals.pop("adjs_truth"), [-1]) # [128,17,17]
                adjs_preds = np.reshape(vals.pop("adjs_preds"), [-1])
                adjs_true_list.extend(adjs_truth)
                adjs_preds_list.extend(adjs_preds)

                batch_size = vals.pop('batch_size')
                avg_meters.add(vals, weight=batch_size)

                '''
                # Writes samples
                tx.utils.write_paired_text(
                    refs.squeeze(), hyps,
                    os.path.join(config.sample_path, 'val.%d'%epoch),
                    append=True, mode='v')
                '''

            except tf.errors.OutOfRangeError:
                logger.info('{}: {}'.format(
                    val_or_test, avg_meters.to_str(precision=4)))
                break

        return avg_meters.avg(), adjs_true_list, adjs_preds_list

    tf.gfile.MakeDirs(config.sample_path)
    tf.gfile.MakeDirs(config.checkpoint_path)

    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        if config.restore:
            logger.info('Restore from: {}'.format(config.restore))
            saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)

        test_true = []
        test_preds = []
        for epoch in range(1, config.max_nepochs+1):
            val_adjs_true = []
            val_adjs_preds = []
            test_adjs_true = []
            test_adjs_preds = []
            train_adjs_true = []
            train_adjs_preds = []

            #logger.info('gamma: {}'.format(gamma_))

            # Train
            iterator.restart_dataset(sess, ['train'])
            train_adjs_true, train_adjs_preds = _train_epoch(sess, epoch, train_adjs_true, train_adjs_preds)

            # Val
            iterator.restart_dataset(sess, 'val')
            _, val_adjs_true, val_adjs_preds = _eval_epoch(sess, epoch, val_adjs_true, val_adjs_preds, 'val')

            saver.save(
                sess, os.path.join(config.checkpoint_path, 'ckpt'), epoch)

            # Test
            iterator.restart_dataset(sess, 'test')
            _, test_adjs_true, test_adjs_preds = _eval_epoch(sess, epoch, test_adjs_true, test_adjs_preds, 'test')

            if epoch == config.max_nepochs:
                test_true = test_adjs_true
                test_preds = test_adjs_preds
            

            #plot_confusion_matrix(train_adjs_true, train_adjs_preds, classes=['1', '0'],
            #          title='Train Confusion matrix, without normalization')
            #plot_confusion_matrix(val_adjs_true, val_adjs_preds, classes=['1', '0'],
            #          title='Val Confusion matrix, without normalization')
        plot_confusion_matrix(test_true, test_preds, classes=np.array(["non-relevant", "relevant"]),
                      normalize=True,title='Test Confusion matrix, without normalization')
        plot_confusion_matrix(test_true, test_preds, classes=np.array(["non-relevant", "relevant"]),
                      normalize=False,title='Test Confusion matrix, without normalization')
        plt.show()


if __name__ == '__main__':
    tf.app.run(main=_main)