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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(5)
import sys
import argparse
import importlib
import pdb
import logging
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
import tensorflow as tf
import texar as tx

from models.baby_model_xlnet import EvolveGTAE
from utils_data.multi_aligned_data_with_numpy import MultiAlignedNumpyData
from utils_train.main_util import *
from utils_preproc import tokenization

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
spiece_model_file = './xlnet_cased_L-12_H-768_A-12/spiece.model'
sp.Load(spiece_model_file)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))



# get config
flags = tf.flags
flags.DEFINE_string('config', 'config_xlnet', 'The config to use.')
flags.DEFINE_string('out', 'out_1024_baby_model_1.1_xlnet_check', 'The config to use.')
flags.DEFINE_string('sample_path', 'samples', 'The config to use.')
flags.DEFINE_string('checkpoint_path', 'checkpoints', 'The config to use.')
flags.DEFINE_string('visloss_train_path', 'vis_train_loss', 'The config to use.')
flags.DEFINE_string('visloss_val_path', 'vis_val_loss', 'The config to use.')
flags.DEFINE_string('visloss_graph_path', 'vis_graph', 'The config to use.')
FLAGS = flags.FLAGS


config = importlib.import_module(FLAGS.config)
output_path = FLAGS.out
sample_path = os.path.join(output_path, FLAGS.sample_path)
checkpoint_path = os.path.join(output_path, FLAGS.checkpoint_path)
vis_train_path = os.path.join(output_path, FLAGS.visloss_train_path)
vis_val_path = os.path.join(output_path, FLAGS.visloss_val_path)
vis_graph_path = os.path.join(output_path, FLAGS.visloss_graph_path)
if output_path == 'none':
    raise ValueError('output path is not specified. E.g. python main.py --out output_path')

if not os.path.exists(output_path):
    os.mkdir(output_path)
    os.mkdir('{}/src'.format(output_path))
    os.system('cp *.py {}/src'.format(output_path))
    os.system('cp models/*.py {}/src'.format(output_path))
    os.system('cp utils_data/*.py {}/src'.format(output_path))
    os.system('cp utils_train/*.py {}/src'.format(output_path))
    os.system('cp models/utils/*.py {}/src'.format(output_path))

# sample_path and checkpoint_path
tf.gfile.MakeDirs(sample_path)
tf.gfile.MakeDirs(checkpoint_path)

# get logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger_format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
logger_format = logging.Formatter(logger_format_str)
logger_sh = logging.StreamHandler()
logger_sh.setFormatter(logger_format)
logger_th = logging.FileHandler('{}.log'.format(os.path.join(output_path, output_path)), mode='w')
logger_th.setFormatter(logger_format)
logger.addHandler(logger_sh)
logger.addHandler(logger_th)

logger.info('config: {}.py'.format(FLAGS.config))

def _main(_):

    # Data
    train_data = MultiAlignedNumpyData(config.train_data)
    val_data = MultiAlignedNumpyData(config.val_data)
    test_data = MultiAlignedNumpyData(config.test_data)
    vocab_size = config.vocab_size
    vocab = {}
    vocab['vocab_size'] = vocab_size
    vocab['bos_token_id'] = config.bos_token_id
    vocab['eos_token_id'] = config.eos_token_id
    ctx_maxSeqLen = config.max_sequence_length_y



    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.FeedableDataIterator(
        {'train_pre': train_data, 'val': val_data, 'test': test_data})
    batch = iterator.get_next()

    # Model
    #global_step = tf.placeholder(tf.int32) #hvd
    lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr') #hvd
    
    
    if config.model_name == 'EvolveGTAE':
        #model = EvolveGTAE(batch, vocab, ctx_maxSeqLen, hvd, global_step, config.model)
        model = EvolveGTAE(batch, vocab, ctx_maxSeqLen, lr, config.model)
    else:
        logger.error('config.model_name: {} is incorrect'.format(config.model_name))
        raise ValueError('config.model_name: {} is incorrect'.format(config.model_name))

    def _train_epoch(sess, lr_,  writer, epoch, flag, verbose=True):
        avg_meters_pre = tx.utils.AverageRecorder(size=10)

        step = 0
        while True:
            try:
                step += 1
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, 'train_pre'),
                    lr:lr_
                }
                vals_pre = sess.run(model.fetches_train_pre, feed_dict=feed_dict)

                merged_summary = vals_pre.pop('merged')

                iteration = (epoch-1)*(config.train_num/config.batch_size)+step
                if iteration % 500 == 0:
                    writer.add_summary(merged_summary, iteration)
                
    

                avg_meters_pre.add(vals_pre)
                
                if verbose and (step == 1 or step % config.display == 0):
                    logger.info('step: {}, {}'.format(step, avg_meters_pre.to_str(4)))
                    sys.stdout.flush()

            except tf.errors.OutOfRangeError:
                logger.info('epoch: {}, {}'.format(epoch, avg_meters_pre.to_str(4)))
                sys.stdout.flush()
                break

    def _eval_epoch(sess, lr_, writer, epoch, val_or_test='val'):##1
        avg_meters = tx.utils.AverageRecorder()
        
        step = 0
        while True:
            try:
                step += 1
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, val_or_test),
                    lr:lr_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }
                vals = sess.run(model.fetches_eval, feed_dict=feed_dict)

                iteration = (epoch-1)*(config.dev_num/config.batch_size)+step
                if val_or_test is 'val' and iteration % 10 == 0:
                    merged_summary = vals['merged']
                    writer.add_summary(merged_summary, iteration)
                vals.pop('merged')
                

                batch_size = vals.pop('batch_size')

                # Computes BLEU
                samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                
                x1x2 = map_ids_to_strs_xlnet(samples['x1x2'], sp, config)
                x1xx2 = map_ids_to_strs_xlnet(samples['x1xx2'], sp, config)

                print("=-----: ", samples['transferred_yy1_pred'])
                hyps_y1 = map_ids_to_strs_xlnet(samples['transferred_yy1_pred'], sp, config)
                refs_y1 = map_ids_to_strs_xlnet(samples['transferred_yy1_gt'], sp, config) ## text == ['a sentence', 'parsed from ids']
                origin_y1 = map_ids_to_strs_xlnet(samples['origin_y1'], sp, config)
                refs_y1 = np.expand_dims(refs_y1, axis=1) #[32,1]
                bleu = tx.evals.corpus_bleu_moses(refs_y1, hyps_y1) #[32]
                vals['bleu_y1'] = bleu

                hyps_y2 = map_ids_to_strs_xlnet(samples['transferred_yy2_pred'], sp, config)
                refs_y2 = map_ids_to_strs_xlnet(samples['transferred_yy2_gt'], sp, config)
                origin_y2 = map_ids_to_strs_xlnet(samples['origin_y2'], sp, config)
                refs_y2 = np.expand_dims(refs_y2, axis=1)
                bleu = tx.evals.corpus_bleu_moses(refs_y2, hyps_y2)
                vals['bleu_y2'] = bleu

                hyps_y3 = map_ids_to_strs_xlnet(samples['transferred_yy3_pred'], sp, config)
                refs_y3 = map_ids_to_strs_xlnet(samples['transferred_yy3_gt'], sp, config)
                origin_y3 = map_ids_to_strs_xlnet(samples['origin_y3'], sp, config)
                refs_y3 = np.expand_dims(refs_y3, axis=1)
                bleu = tx.evals.corpus_bleu_moses(refs_y3, hyps_y3)
                vals['bleu_y3'] = bleu

                avg_meters.add(vals, weight=batch_size)

                
                # Writes samples
                if val_or_test is 'test':
                    tx.utils.write_paired_text(
                        x1x2, x1xx2,
                        os.path.join(sample_path, 'val_x.%d'%epoch),
                        append=True, mode='v')
                    tx.utils.write_paired_text(
                        refs_y1.squeeze(), hyps_y1,
                        os.path.join(sample_path, 'val_y1.%d'%epoch),
                        append=True, mode='v')
                    tx.utils.write_paired_text(
                        refs_y2.squeeze(), hyps_y2,
                        os.path.join(sample_path, 'val_y2.%d'%epoch),
                        append=True, mode='v')
                    tx.utils.write_paired_text(
                        refs_y3.squeeze(), hyps_y3,
                        os.path.join(sample_path, 'val_y3.%d'%epoch),
                        append=True, mode='v')

                    tx.utils.write_paired_text(
                        refs_y1.squeeze(), origin_y1,
                        os.path.join(sample_path, 'val_yy1gt_y1.%d'%epoch),
                        append=True, mode='v')
                    tx.utils.write_paired_text(
                        refs_y2.squeeze(), origin_y2,
                        os.path.join(sample_path, 'val_yy2gt_y2.%d'%epoch),
                        append=True, mode='v')
                    tx.utils.write_paired_text(
                        refs_y3.squeeze(), origin_y3,
                        os.path.join(sample_path, 'val_yy3gt_y3.%d'%epoch),
                        append=True, mode='v')

            except tf.errors.OutOfRangeError:
                logger.info('{}: {}'.format(
                    val_or_test, avg_meters.to_str(precision=4)))
                break

        return avg_meters.avg()


    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # visulization
        graph_writer=tf.summary.FileWriter(vis_graph_path, sess.graph)

        # visulization
        train_writer = tf.summary.FileWriter(vis_train_path, tf.Graph())
        # visulization
        val_writer = tf.summary.FileWriter(vis_val_path, tf.Graph())

        
        saver = tf.train.Saver(max_to_keep=None)

        if config.restore:
            logger.info('Restore from: {}'.format(config.restore))
            saver.restore(sess, config.restore)

        iterator.initialize_dataset(sess)
        lr_ = config.initial_lr
    
        for epoch in range(0, config.max_nepochs + 1):
            flag = True
            if epoch<=3:
                lr_=config.initial_lr*(epoch+1)
            if epoch>=10 and epoch %2==0:
                lr_*=0.25

            logger.info('learning rate: {}'.format(lr_))
            # Train
            iterator.restart_dataset(sess, ['train_pre'])
            
            _train_epoch(sess, lr_, train_writer, epoch, flag)

            # Val
            iterator.restart_dataset(sess, 'val')
            _eval_epoch(sess, lr_,val_writer, epoch, 'val') ##1

            if epoch%4==0:
                saver.save(
                    sess, os.path.join(checkpoint_path, 'ckpt'), epoch)

            # Test
            iterator.restart_dataset(sess, 'test')
            _eval_epoch(sess, lr_,val_writer, epoch, 'test')

            

        graph_writer.close()
        train_writer.close()
        val_writer.close()


if __name__ == '__main__':
    tf.app.run(main=_main)