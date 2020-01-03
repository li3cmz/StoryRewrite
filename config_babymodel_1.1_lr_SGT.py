"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy
import texar as tx
import tensorflow as tf

initial_lr = 5e-4
max_nepochs = 20 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder) ###modify
display = 500  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).
restore = './out_1020_baby_model_1.1_addSGT/checkpoints/ckpt-15'#'./out_0919_main_v24-005-002/checkpoints/ckpt-5'  ###modify # Model snapshot to restore from

model_name = 'EvolveGTAE'

lambda_adj_final = 1 # Weight of the adj_final loss
lambda_rephraser = 1
lambda_adj_cft = 1

max_sequence_length_y = 128 # Maximum number of tokens in a sentence ###check
max_sequence_length_ctx = 128
max_decode_yy = 128  ###maybe can't be so big

batch_size=8
max_utterance_cnt=2
vocab_size = 30522
bos_token_id = 101
eos_token_id = 102
pad_id = 0

distributed = False
vocab_file = './bert_pretrained_models/bert_pretrained_models/uncased_L-12_H-768_A-12/vocab.txt'
do_lower_case = True

## Data configs
feature_original_types = {
    # Reading features from TFRecord data file.
    # E.g., Reading feature "text_ids" as dtype `tf.int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "input_ids_x1x2ysx1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "input_mask_x1x2ysx1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "segment_ids_x1x2ysx1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "input_ids_x1x2ysx1xx2yy": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "input_mask_x1x2ysx1xx2yy": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "segment_ids_x1x2ysx1xx2yy": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],

    "input_ids_x1x2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "input_mask_x1x2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "segment_ids_x1x2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],

    "input_ids_x1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "input_mask_x1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],
    "segment_ids_x1xx2": ["tf.int64", "FixedLenFeature", max_sequence_length_ctx],

    "input_ids_y1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_y1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_y1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_ids_y2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_y2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_y2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_ids_y3": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_y3": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_y3": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    
    "input_ids_yy1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_yy1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_yy1": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_ids_yy2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_yy2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_yy2": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_ids_yy3": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "input_mask_yy3": ["tf.int64", "FixedLenFeature", max_sequence_length_y],
    "segment_ids_yy3": ["tf.int64", "FixedLenFeature", max_sequence_length_y]
}
feature_convert_types = {
    # Converting feature dtype after reading. E.g.,
    # Converting the dtype of feature "text_ids" from `tf.int64` (as above)
    # to `tf.int32`
    "input_ids_x1x2ysx1xx2": "tf.int32",
    "input_mask_x1x2ysx1xx2": "tf.int32",
    "segment_ids_x1x2ysx1xx2": "tf.int32",
    "input_ids_x1x2ysx1xx2yy": "tf.int32",
    "input_mask_x1x2ysx1xx2yy": "tf.int32",
    "segment_ids_x1x2ysx1xx2yy": "tf.int32",

    "input_ids_x1x2": "tf.int32",
    "input_mask_x1x2": "tf.int32",
    "segment_ids_x1x2": "tf.int32",

    "input_ids_x1xx2": "tf.int32",
    "input_mask_x1xx2": "tf.int32",
    "segment_ids_x1xx2": "tf.int32",

    "input_ids_y1": "tf.int32",
    "input_mask_y1": "tf.int32",
    "segment_ids_y1": "tf.int32",
    "input_ids_y2": "tf.int32",
    "input_mask_y2": "tf.int32",
    "segment_ids_y2": "tf.int32",
    "input_ids_y3": "tf.int32",
    "input_mask_y3": "tf.int32",
    "segment_ids_y3": "tf.int32",

    "input_ids_yy1": "tf.int32",
    "input_mask_yy1": "tf.int32",
    "segment_ids_yy1": "tf.int32",
    "input_ids_yy2": "tf.int32",
    "input_mask_yy2": "tf.int32",
    "segment_ids_yy2": "tf.int32",
    "input_ids_yy3": "tf.int32",
    "input_mask_yy3": "tf.int32",
    "segment_ids_yy3": "tf.int32"
}


mini="" ###modify #"/mini" or ""
text_data_dir = "./data/TimeTravel/bert3{}".format(mini) ###modify bert or bert2
adj_data_dir = "./data/TimeTravel{}".format(mini)
train_data = {
    'batch_size': batch_size,
    #'seed': 123,
    'datasets': [
        {
            "files": "{}/train_supervised_large.tf_record".format(text_data_dir),
            "data_name": "text",
            'data_type': 'tf_record',
            "feature_original_types": feature_original_types,
            "feature_convert_types": feature_convert_types,
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y1_adjs_dirt.tfrecords'.format(adj_data_dir), ###modify
            'data_type': 'tf_record',
            'data_name': 'y1_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y2_adjs_dirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'y2_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y3_adjs_dirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'y3_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy1_adjs_dirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy1_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy2_adjs_dirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy2_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy3_adjs_dirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy3_d',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y1_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'y1_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y2_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'y2_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_y3_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'y3_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy1_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy1_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy2_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy2_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        },
        {
            'files': '{}/TimeTravel.train_supervised_large_yy3_adjs_undirt.tfrecords'.format(adj_data_dir),
            'data_type': 'tf_record',
            'data_name': 'yy3_und',
            'numpy_options': {
                'numpy_ndarray_name': 'adjs',
                'shape': [max_sequence_length_y + 2, max_sequence_length_y + 2],
                'dtype': 'tf.int32'
            },
            'feature_original_types':{
                'adjs':['tf.string', 'FixedLenFeature']
            }
        }
    ],
    'name': 'train',
    'shuffle':True,
    "shuffle_buffer_size": 1000
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = '{}/dev_data.tf_record'.format(text_data_dir)
val_data['datasets'][1]['files'] = '{}/TimeTravel.dev_data_y1_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][2]['files'] = '{}/TimeTravel.dev_data_y2_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][3]['files'] = '{}/TimeTravel.dev_data_y3_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][4]['files'] = '{}/TimeTravel.dev_data_yy1_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][5]['files'] = '{}/TimeTravel.dev_data_yy2_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][6]['files'] = '{}/TimeTravel.dev_data_yy3_adjs_dirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][7]['files'] = '{}/TimeTravel.dev_data_y1_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][8]['files'] = '{}/TimeTravel.dev_data_y2_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][9]['files'] = '{}/TimeTravel.dev_data_y3_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][10]['files'] = '{}/TimeTravel.dev_data_yy1_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][11]['files'] = '{}/TimeTravel.dev_data_yy2_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['datasets'][12]['files'] = '{}/TimeTravel.dev_data_yy3_adjs_undirt.tfrecords'.format(adj_data_dir)
val_data['shuffle'] = False

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = '{}/test_data.tf_record'.format(text_data_dir)
test_data['datasets'][1]['files'] = '{}/TimeTravel.test_data_y1_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][2]['files'] = '{}/TimeTravel.test_data_y2_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][3]['files'] = '{}/TimeTravel.test_data_y3_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][4]['files'] = '{}/TimeTravel.test_data_yy1_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][5]['files'] = '{}/TimeTravel.test_data_yy2_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][6]['files'] = '{}/TimeTravel.test_data_yy3_adjs_dirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][7]['files'] = '{}/TimeTravel.test_data_y1_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][8]['files'] = '{}/TimeTravel.test_data_y2_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][9]['files'] = '{}/TimeTravel.test_data_y3_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][10]['files'] = '{}/TimeTravel.test_data_yy1_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][11]['files'] = '{}/TimeTravel.test_data_yy2_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['datasets'][12]['files'] = '{}/TimeTravel.test_data_yy3_adjs_undirt.tfrecords'.format(adj_data_dir)
test_data['shuffle'] = False


dim_hidden = 768
dim_hidden_mini = 512
model = {
    'gpt2_hidden_dim':dim_hidden,
    'dim_c_big': dim_hidden,
    'dim_c': dim_hidden,
    'wordEmbedder':{
        'dim':dim_hidden
    },
    'gpt2_posEmbedder':{
        'dim':dim_hidden
    },
    'bert_encoder':{
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
        'num_blocks': 2,
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
    'transformer_decoderToEncoder':{
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
        }
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
        'max_decoding_length_train': max_decode_yy,
        'max_decoding_length_infer': max_decode_yy,
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
