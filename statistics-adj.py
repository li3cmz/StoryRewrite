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
"""Text style transfer Under Linguistic Constraints
This is the implementation of:
Linguistic-Constrained Text Style Transfer for Content and Logic Preservation
Follow the instructions in README.md to run the code
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import sys
import argparse
import importlib
import pdb
import logging
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
import tensorflow as tf
import texar as tx

from utils_data.multi_aligned_data_with_numpy import MultiAlignedNumpyData
from utils_train.main_util import *                                                     ###modify
from utils_preproc import tokenization
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(6)                                                ###modify

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


# get config
flags = tf.flags
flags.DEFINE_string('config', 'config_storyRw', 'The config to use.')                              ###modify
flags.DEFINE_string('out', 'out_statistics_adj', 'The config to use.')                      ###modify
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
    '''
    if config.distributed:
        import horovod.tensorflow as hvd
        hvd.init()
        for i in range(14):
            config.train_data["datasets"][i]["num_shards"] = hvd.size()
            config.train_data["datasets"][i]["shard_id"] = hvd.rank()
        config.train_data["batch_size"] //= hvd.size()
    '''

    # Data
    train_data = MultiAlignedNumpyData(config.train_data)
    val_data = MultiAlignedNumpyData(config.val_data)
    test_data = MultiAlignedNumpyData(config.test_data)
    vocab = train_data.vocab(0)
    ctx_maxSeqLen = config.max_sequence_length

    
    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.FeedableDataIterator(
        {'train_pre': train_data, 'val': val_data, 'test': test_data})
    batch = iterator.get_next()

    node_add_1_sum = 0
    node_add_2_sum = 0
    node_add_3_sum = 0
    node_dec_1_sum = 0
    node_dec_2_sum = 0
    node_dec_3_sum = 0

    edge_add_1_sum = 0
    edge_add_2_sum = 0
    edge_add_3_sum = 0
    edge_dec_1_sum = 0
    edge_dec_2_sum = 0
    edge_dec_3_sum = 0

    edge_C_1_sum = 0
    edge_C_2_sum = 0
    edge_C_3_sum = 0

    samples_num = 0
    
    # Runs the logics
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        iterator.initialize_dataset(sess)

        for mode in ['test']:#, 'val', 'test']:
            #train
            iterator.restart_dataset(sess, [mode])
            while True:
                try:
                    
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, mode),
                    }
                    fetches_data = {
                        'batch': batch,
                    }
                    inputs = sess.run(fetches_data, feed_dict)['batch']
                    
                    
                    text_ids_y1 = inputs['y1_yy1_text_ids'][:,0,:]   # [batch, maxlen1]
                    text_ids_yy1 = inputs['y1_yy1_text_ids'][:,1,:]  # [batch, maxlen1]
                    text_ids_y2 = inputs['y2_yy2_text_ids'][:,0,:]   # [batch, maxlen2]
                    text_ids_yy2 = inputs['y2_yy2_text_ids'][:,1,:]  # [batch, maxlen2]
                    text_ids_y3 = inputs['y3_yy3_text_ids'][:,0,:]   # [batch, maxlen3]
                    text_ids_yy3 = inputs['y3_yy3_text_ids'][:,1,:]  # [batch, maxlen3]

                    sequence_length_y1 = inputs['y1_yy1_length'][:,0]
                    sequence_length_y2 = inputs['y2_yy2_length'][:,0]
                    sequence_length_y3 = inputs['y3_yy3_length'][:,0]
                    sequence_length_yy1 = inputs['y1_yy1_length'][:,1]
                    sequence_length_yy2 = inputs['y2_yy2_length'][:,1]
                    sequence_length_yy3 = inputs['y3_yy3_length'][:,1]

                    enc_shape_y1 = tf.shape(text_ids_y1)[1]
                    enc_shape_y2 = tf.shape(text_ids_y2)[1]
                    enc_shape_y3 = tf.shape(text_ids_y3)[1]

                    adjs_y1_undirt = np.int32(np.reshape(inputs['y1_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    adjs_y2_undirt = np.int32(np.reshape(inputs['y2_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    adjs_y3_undirt = np.int32(np.reshape(inputs['y3_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    adjs_yy1_undirt = np.int32(np.reshape(inputs['yy1_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    adjs_yy2_undirt = np.int32(np.reshape(inputs['yy2_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    adjs_yy3_undirt = np.int32(np.reshape(inputs['yy3_und_adjs'], [-1,ctx_maxSeqLen+2,ctx_maxSeqLen+2]))
                    #print(np.shape(adjs_yy3_undirt))
                    

                    
                    
                    node_add_1 = np.sum(np.where((sequence_length_yy1 - sequence_length_y1)>0, (sequence_length_yy1 - sequence_length_y1), 0))
                    node_dec_1 = np.sum(np.where((sequence_length_y1 - sequence_length_yy1)>0, (sequence_length_y1 - sequence_length_yy1), 0))
                    node_add_2 = np.sum(np.where((sequence_length_yy2 - sequence_length_y2)>0, (sequence_length_yy2 - sequence_length_y2), 0))
                    node_dec_2 = np.sum(np.where((sequence_length_y2 - sequence_length_yy2)>0, (sequence_length_y2 - sequence_length_yy2), 0))
                    node_add_3 = np.sum(np.where((sequence_length_yy3 - sequence_length_y3)>0, (sequence_length_yy3 - sequence_length_y3), 0))
                    node_dec_3 = np.sum(np.where((sequence_length_y3 - sequence_length_yy3)>0, (sequence_length_y3 - sequence_length_yy3), 0))

                    
                    node_add_1_sum+=node_add_1
                    node_add_2_sum+=node_add_2
                    node_add_3_sum+=node_add_3
                    node_dec_1_sum+=node_dec_1
                    node_dec_2_sum+=node_dec_2
                    node_dec_3_sum+=node_dec_3

                    edge_add_1 = np.sum(cal_edge_2(adjs_y1_undirt*(-1) + adjs_yy1_undirt)[0])
                    edge_add_2 = np.sum(cal_edge_2(adjs_y2_undirt*(-1) + adjs_yy2_undirt)[0])
                    edge_add_3 = np.sum(cal_edge_2(adjs_y3_undirt*(-1) + adjs_yy3_undirt)[0])
                    edge_dec_1 = np.sum(cal_edge_2(adjs_y1_undirt*(-1) + adjs_yy1_undirt)[1])
                    edge_dec_2 = np.sum(cal_edge_2(adjs_y2_undirt*(-1) + adjs_yy2_undirt)[1])
                    edge_dec_3 = np.sum(cal_edge_2(adjs_y3_undirt*(-1) + adjs_yy3_undirt)[1])
                    edge_C_1 = np.sum(cal_edge(np.abs(adjs_y1_undirt*(-1) + adjs_yy1_undirt)))
                    edge_C_2 = np.sum(cal_edge(np.abs(adjs_y2_undirt*(-1) + adjs_yy2_undirt)))
                    edge_C_3 = np.sum(cal_edge(np.abs(adjs_y3_undirt*(-1) + adjs_yy3_undirt)))

                    edge_C_1_sum += edge_C_1
                    edge_C_2_sum += edge_C_2
                    edge_C_3_sum += edge_C_3
                    
                    edge_add_1_sum+=edge_add_1
                    edge_add_2_sum+=edge_add_2
                    edge_add_3_sum+=edge_add_3

                    edge_dec_1_sum+=edge_dec_1
                    edge_dec_2_sum+=edge_dec_2
                    edge_dec_3_sum+=edge_dec_3

                    samples_num+=np.shape(adjs_y1_undirt)[0] #batch size

                    print("samples_num: ",samples_num)#, tx.utils.map_ids_to_strs(text_ids_y1, vocab), '---------', tx.utils.map_ids_to_strs(text_ids_yy1, vocab))
                    #print((-1)*adjs_y1_undirt,'\n')
                    #print(adjs_yy1_undirt, '\n')
                    #print((-1)*adjs_y1_undirt+adjs_yy1_undirt, '\n')
                    print("edge_num_y1: ", cal_edge(adjs_y1_undirt), "edge_num_yy1: ", cal_edge(adjs_yy1_undirt))
                    
                    print("node_add_1: ", node_add_1_sum/samples_num, "node_dec_1: ", node_dec_1_sum/samples_num)
                    print("edge_add_1: ", edge_add_1_sum/samples_num, "edge_dec_1: ", edge_dec_1_sum/samples_num)
                    print("edge_C_1_sum: ", edge_C_1_sum/samples_num)
                    print('\n')
                except tf.errors.OutOfRangeError:
                    break
    
    avg_node_add_1 = node_add_1_sum / float(samples_num)
    avg_node_add_2 = node_add_2_sum / float(samples_num)
    avg_node_add_3 = node_add_3_sum / float(samples_num)
    
    avg_node_dec_1 = node_dec_1_sum / float(samples_num)
    avg_node_dec_2 = node_dec_2_sum / float(samples_num)
    avg_node_dec_3 = node_dec_3_sum / float(samples_num)


    avg_edge_add_1 = edge_add_1_sum / float(samples_num)
    avg_edge_add_2 = edge_add_2_sum / float(samples_num)
    avg_edge_add_3 = edge_add_3_sum / float(samples_num)
    
    avg_edge_dec_1 = edge_dec_1_sum / float(samples_num)
    avg_edge_dec_2 = edge_dec_2_sum / float(samples_num)
    avg_edge_dec_3 = edge_dec_3_sum / float(samples_num)

    avg_edge_C_1 = edge_C_1_sum / float(samples_num)
    avg_edge_C_2 = edge_C_2_sum / float(samples_num)
    avg_edge_C_3 = edge_C_3_sum / float(samples_num)
    
    print("avg_node_add_1: ", avg_node_add_1, "avg_node_dec_1: ", avg_node_dec_1, "avg_edge_add_1: ", avg_edge_add_1, "avg_edge_dec_1: ", avg_edge_dec_1)
    print("avg_node_add_2: ", avg_node_add_2, "avg_node_dec_2: ", avg_node_dec_2, "avg_edge_add_2: ", avg_edge_add_2, "avg_edge_dec_2: ", avg_edge_dec_2)
    print("avg_node_add_3: ", avg_node_add_3, "avg_node_dec_3: ", avg_node_dec_3, "avg_edge_add_3: ", avg_edge_add_3, "avg_edge_dec_3: ", avg_edge_dec_3)
    print("avg_edge_C_1: ", avg_edge_C_1, "avg_edge_C_2: ", avg_edge_C_2, "avg_edge_C_3: ", avg_edge_C_3)

def cal_edge(adjs):
    edge_num_list = []
    for adj in adjs:
        edge_num = np.sum(adj)
        edge_num_list.append(edge_num)
    
    return np.array(edge_num_list) #[batch]

def cal_edge_2(adjs):
    edge_num_add_list = []
    edge_num_dec_list = []
    for adj in adjs:
        dec = 0
        add = 0
        for i in range(np.shape(adj)[0]):
            for j in range(np.shape(adj)[1]):
                if adj[i][j] == -1:
                    dec+=1
                elif adj[i][j] == 1:
                    add+=1
        edge_num_add_list.append(add)
        edge_num_dec_list.append(dec)
    return np.array(edge_num_add_list), np.array(edge_num_dec_list)

if __name__ == '__main__':
    tf.app.run(main=_main)