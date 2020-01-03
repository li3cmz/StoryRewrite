import json
import numpy as np 
import tensorflow as tf
import time
import argparse
from tqdm import tqdm

'''
def _parse_function(example_proto):
    features = {
            'adjs': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    data = tf.decode_raw(parsed_features['adjs'], tf.int32)
    return data
'''
'''
"input_ids_x1xx2": tf.FixedLenFeature([], tf.int64),
        "input_mask_x1xx2": tf.FixedLenFeature([], tf.int64),
        "segment_ids_x1xx2": tf.FixedLenFeature([], tf.int64),
        "input_mask_x1x2": tf.FixedLenFeature([], tf.int64),
        "segment_ids_x1x2": tf.FixedLenFeature([], tf.int64),
'''
max_sequence_length_x = 33
def _parse_function(example_proto):
    features = {
        'input_ids_x1x2': tf.FixedLenFeature([33], tf.int64),
        "segment_ids_x1xx2": tf.FixedLenFeature([33], tf.int64),
        "input_mask_x1x2": tf.FixedLenFeature([33], tf.int64),
        "segment_ids_x1x2": tf.FixedLenFeature([33], tf.int64),
    
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    data1 = parsed_features#tf.decode_raw(parsed_features['input_ids_x1x2'], tf.int32)
    return data1

def load_tfrecords(srcfile):
    sess = tf.Session()

    dataset = tf.data.TFRecordDataset(srcfile)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(64)

    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    while True:
        try:
            data = sess.run(next_data)
            for each in data:
                print(each)
            #print(np.size(data))
            time.sleep(3)
        except tf.errors.OutOfRangeError:
            break


tf_name = "../data/TimeTravel/bert/{}".format('dev_data_x.tf_record')

load_tfrecords(tf_name)