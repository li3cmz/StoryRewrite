"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy
import texar as tx
import tensorflow as tf

initial_lr = 1e-4
max_nepochs = 50 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 2 # Number of pre-train epochs (training as autoencoder) ###modify
display = 20  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).

restore =''###modify # Model snapshot to restore from

model_name = 'EvolveGTAE'

lambda_adj_final = 1 # Weight of the adj_final loss
lambda_rephraser = 1
lambda_adj_cft = 1

max_sequence_length = 128 # Maximum number of tokens in a sentence ###check  不包括BOS 和 EOS

max_utterance_cnt = 2
max_utterance_cnt_ctx = 8 
#x1x2|||x1xx2|||x1x2yx1xx2|||x1x2yx1xx2yy|||x1x2y|||x1|||x1x2yx1my|||x1x2yx1m
# 一个单元的单个句子之间用<SEP>分隔
mode_name='./data/TimeTravel/baby_model5' ###modify

batch_size = 32
train_num = 28167
dev_num = 1856
test_num = 1854

train_data = {
    'batch_size': batch_size,
    #'seed': 123,
    'datasets': [
        {
            'files': '{}/train_supervised_large_ctx.text'.format(mode_name), #首先这里给的句子的maxlen要<=128, 当小于128，会pad到128，因为这样才能做model里面的一些操作
            'vocab_file': './data/TimeTravel/vocab2', ###modify
            'data_name': 'ctx',
            "variable_utterance": True,
            "max_utterance_cnt": max_utterance_cnt_ctx,
            "max_seq_length":max_sequence_length,
            "pad_to_max_seq_length": True,
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y1_yy1.text',
            'vocab_file': './data/TimeTravel/vocab2', ###modify
            'data_name': 'y1_yy1',
            "variable_utterance": True,
            "max_utterance_cnt": max_utterance_cnt,
            "max_seq_length":max_sequence_length,
            "pad_to_max_seq_length": True,
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y2_yy2.text',
            'vocab_file': './data/TimeTravel/vocab2', 
            'data_name': 'y2_yy2',
            "variable_utterance": True,
            "max_utterance_cnt": max_utterance_cnt,
            "max_seq_length":max_sequence_length,
            "pad_to_max_seq_length": True,
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y3_yy3.text',
            'vocab_file': './data/TimeTravel/vocab2', 
            'data_name': 'y3_yy3',
            "variable_utterance": True,
            "max_utterance_cnt": max_utterance_cnt,
            "max_seq_length":max_sequence_length,
            "pad_to_max_seq_length": True,
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y1_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y1_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y2_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y2_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y3_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y3_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy1_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy1_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy2_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy2_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy3_adjs_dirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy3_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y1_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y1_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y2_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y2_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_y3_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'y3_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy1_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy1_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy2_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy2_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': './data/TimeTravel/TimeTravel.train_supervised_large_yy3_adjs_undirt.tfrecords',
            'data_type': 'tf_record',
            'data_name': 'yy3_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length + 2, max_sequence_length + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        }
    ],
    'name': 'train'
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = '{}/dev_data_ctx.text'.format(mode_name)
val_data['datasets'][1]['files'] = './data/TimeTravel/TimeTravel.dev_data_y1_yy1.text'
val_data['datasets'][2]['files'] = './data/TimeTravel/TimeTravel.dev_data_y2_yy2.text'
val_data['datasets'][3]['files'] = './data/TimeTravel/TimeTravel.dev_data_y3_yy3.text'
val_data['datasets'][4]['files'] = './data/TimeTravel/TimeTravel.dev_data_y1_adjs_dirt.tfrecords'
val_data['datasets'][5]['files'] = './data/TimeTravel/TimeTravel.dev_data_y2_adjs_dirt.tfrecords'
val_data['datasets'][6]['files'] = './data/TimeTravel/TimeTravel.dev_data_y3_adjs_dirt.tfrecords'
val_data['datasets'][7]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy1_adjs_dirt.tfrecords'
val_data['datasets'][8]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy2_adjs_dirt.tfrecords'
val_data['datasets'][9]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy3_adjs_dirt.tfrecords'
val_data['datasets'][10]['files'] = './data/TimeTravel/TimeTravel.dev_data_y1_adjs_undirt.tfrecords'
val_data['datasets'][11]['files'] = './data/TimeTravel/TimeTravel.dev_data_y2_adjs_undirt.tfrecords'
val_data['datasets'][12]['files'] = './data/TimeTravel/TimeTravel.dev_data_y3_adjs_undirt.tfrecords'
val_data['datasets'][13]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy1_adjs_undirt.tfrecords'
val_data['datasets'][14]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy2_adjs_undirt.tfrecords'
val_data['datasets'][15]['files'] = './data/TimeTravel/TimeTravel.dev_data_yy3_adjs_undirt.tfrecords'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = '{}/test_data_ctx.text'.format(mode_name)
test_data['datasets'][1]['files'] = './data/TimeTravel/TimeTravel.test_data_y1_yy1.text'
test_data['datasets'][2]['files'] = './data/TimeTravel/TimeTravel.test_data_y2_yy2.text'
test_data['datasets'][3]['files'] = './data/TimeTravel/TimeTravel.test_data_y3_yy3.text'
test_data['datasets'][4]['files'] = './data/TimeTravel/TimeTravel.test_data_y1_adjs_dirt.tfrecords'
test_data['datasets'][5]['files'] = './data/TimeTravel/TimeTravel.test_data_y2_adjs_dirt.tfrecords'
test_data['datasets'][6]['files'] = './data/TimeTravel/TimeTravel.test_data_y3_adjs_dirt.tfrecords'
test_data['datasets'][7]['files'] = './data/TimeTravel/TimeTravel.test_data_yy1_adjs_dirt.tfrecords'
test_data['datasets'][8]['files'] = './data/TimeTravel/TimeTravel.test_data_yy2_adjs_dirt.tfrecords'
test_data['datasets'][9]['files'] = './data/TimeTravel/TimeTravel.test_data_yy3_adjs_dirt.tfrecords'
test_data['datasets'][10]['files'] = './data/TimeTravel/TimeTravel.test_data_y1_adjs_undirt.tfrecords'
test_data['datasets'][11]['files'] = './data/TimeTravel/TimeTravel.test_data_y2_adjs_undirt.tfrecords'
test_data['datasets'][12]['files'] = './data/TimeTravel/TimeTravel.test_data_y3_adjs_undirt.tfrecords'
test_data['datasets'][13]['files'] = './data/TimeTravel/TimeTravel.test_data_yy1_adjs_undirt.tfrecords'
test_data['datasets'][14]['files'] = './data/TimeTravel/TimeTravel.test_data_yy2_adjs_undirt.tfrecords'
test_data['datasets'][15]['files'] = './data/TimeTravel/TimeTravel.test_data_yy3_adjs_undirt.tfrecords'



dim_hidden = 512#768
dim_hidden_mini = 512
model = {
    'gpt2_hidden_dim':dim_hidden,
    'dim_c_big': dim_hidden,
    'dim_c': dim_hidden,
    'wordEmbedder':{
        'dim':dim_hidden
    },
    'prelu':{
        "dim": dim_hidden,
        "name": "prelu",
    },
    'gpt2_posEmbedder':{
        'dim':dim_hidden
    },
    'bert_encoder':{
        'pretrained_model_name': None,
        # default params are enough
    },
    'bidirectionalRNNEncoder':{
        "rnn_cell_fw": {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden/2, ###check Q
            },
        },
        "rnn_cell_share_config": True,
        "output_layer_fw": {
            "num_layers": 0,
            "layer_size": 128,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": False
        },
        "output_layer_bw": {
            # Same hyperparams and default values as "output_layer_fw"
            # ...
        },
        "output_layer_share_config": True,
        "name": "bidirectional_rnn_encoder"
    },
    'unidirectionalRNNEncoder':{
        "rnn_cell": {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden, ###check Q  ###debug
            },
        },
        "output_layer": {
            "num_layers": 0,
            "layer_size": 128,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": False
        },
        "name": "unidirectional_rnn_encoder"
    },
    'EmbeddingNormalize':{
        "name": "EmbeddingNormalize"
    },
    'EmbeddingNormalizeNN1':{
        "name": "EmbeddingNormalizeNN",
        "size":dim_hidden,
        "epsilon": 1e-3,
        "decay":0.99,
        "name_scope":'nnBN1'
    },
    'EmbeddingNormalizeNN2':{
        "name": "EmbeddingNormalizeNN",
        "size":dim_hidden,
        "epsilon": 1e-3,
        "decay":0.99,
        "name_scope":'nnBN2'
    },
    'encoder': {
        'num_blocks': 2,
        'dim': dim_hidden,
        'use_bert_config': False,
        'embedding_dropout': 0.1,
        'residual_dropout': 0.1,
        'graph_multihead_attention': {
            'name': 'multihead_attention',
            'num_units': dim_hidden,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'output_dim': dim_hidden,
            'use_bias': False,
        },
        'initializer': None,
        'poswise_feedforward': {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": dim_hidden*4, ###debug
                        "activation": "relu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "rate": 0.1,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": dim_hidden,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn"
        },
        'name': 'graph_transformer_encoder',
    },
    'cross_graph_encoder': {
        'num_blocks': 2,
        'dim': dim_hidden,
        'use_bert_config': False,
        'use_adj':False,
        'embedding_dropout': 0.1,
        'residual_dropout': 0.1,
        'graph_multihead_attention': {
            'name': 'multihead_attention',
            'num_units': dim_hidden,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'output_dim': dim_hidden,
            'use_bias': False,
        },
        'initializer': None,
        'poswise_feedforward': {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": dim_hidden*4, ###debug
                        "activation": "relu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "rate": 0.1,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": dim_hidden,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn"
        },
        'name': 'cross_graph_encoder',
    },
    'adjMultiheadAttention_encoder':{
        'initializer': None,
        'num_heads': 8,
        'output_dim': 1,
        'num_units': 512,
        'dropout_rate': 0.1,
        'use_bias': True,
        "name": "adjMultiheadAttention_encoder",
    },
    'transformer_encoder':{
        'num_blocks': 1,
        'dim': dim_hidden,  ###debug
        'use_bert_config': False,
        'embedding_dropout': 0.1,
        'residual_dropout': 0.1,
        'multihead_attention':{
            'name': 'multihead_attention',
            'num_units': dim_hidden,
            'output_dim': dim_hidden,
            'num_heads': 8,
            'dropout_rate': 0.1,
            'output_dim': dim_hidden,
            'use_bias': False,
        },
        'initializer': None,
        'poswise_feedforward': {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": dim_hidden*4,
                        "activation": "relu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dropout",
                    "kwargs": {
                        "rate": 0.1,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": dim_hidden,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn"
        },
        'name': 'transformer_encoder',
    },
    'transformer_decoder':{
        "name": "transformer_decoder"
    },
    'pooling_aggregator':{
        "output_dim": dim_hidden,
        "input_dim":dim_hidden,
        "concat": True,
        "pooling":'meanpooling',
        "dropout_rate":0.0,
        "l2_reg":0.1,
        "use_bias":True,
        "activation":tf.nn.relu,
        "seed":1024,
        "update_weights":False,
        "name": "pooling_aggregator",
    },
    'gpt2_config_decoder':{
        'dim': dim_hidden,
        'num_blocks': 12,
        'multihead_attention': {
            'use_bias': True,
            'num_units': dim_hidden,
            'num_heads': 12,
            'output_dim': dim_hidden,
        },
        'initializer': {
            'type': 'variance_scaling_initializer',
            'kwargs': {
                'scale': 1.0,
                'mode': 'fan_avg',
                'distribution': 'uniform',
            },
        },
        'poswise_feedforward': {
            "layers": [
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv1",
                        "units": dim_hidden*4,
                        "activation": "gelu",
                        "use_bias": True,
                    }
                },
                {
                    "type": "Dense",
                    "kwargs": {
                        "name": "conv2",
                        "units": dim_hidden,
                        "use_bias": True,
                    }
                }
            ],
            "name": "ffn",
        },
    },
    'rephrase_encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'rephrase_decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': dim_hidden,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'DynamicBahdanauAttention',
            'kwargs': {
                'num_units': dim_hidden,
            },
            'attention_layer_size': dim_hidden,
        },
        'max_decoding_length_train': max_sequence_length,
        'max_decoding_length_infer': max_sequence_length,
    },
    'opt': {
        "optimizer": {
            "type": "AdamOptimizer",
            "kwargs": {
                "learning_rate": 1e-4,
            }
        },
        "learning_rate_decay": {
            "type": "",
            "kwargs": {},
            "min_learning_rate": 1e-7,
            "start_decay_step": 5*(train_num/batch_size),
            "end_decay_step": 50*(train_num/batch_size)
        },
        "gradient_clip": {
            "type": "",
            "kwargs": {}
        },
        "gradient_noise_scale": None,
        "name": None
    },
}
