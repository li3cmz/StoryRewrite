# Copyright 2019 The Texar Authors. All Rights Reserved.
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
"""
Utils of data preprocessing for GPT2 training.
"""

import os
import collections
import csv
import tensorflow as tf

# pylint: disable=invalid-name, too-many-arguments

class InputExample(object):

    def __init__(self, x1, x2, xx2, y1, y2, y3, y, yy1_end1, yy2_end1, yy3_end1, yy1_end2=None, yy2_end2=None, yy3_end2=None, yy1_end3=None, yy2_end3=None, yy3_end3=None):
        self.x1 = x1
        self.x2 = x2
        self.xx2 = xx2
        self.y = y
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

        self.yy1_end1=yy1_end1
        self.yy2_end1=yy2_end1
        self.yy3_end1=yy3_end1
        if yy1_end2 is not None:
            self.yy1_end2=yy1_end2
            self.yy2_end2=yy2_end2
            self.yy3_end2=yy3_end2
            self.yy1_end3=yy1_end3
            self.yy2_end3=yy2_end3
            self.yy3_end3=yy3_end3



class InputFeatures(object):

    def __init__(self, x1x2yx1xx2, x1x2yx1my, x1x2yx1xx2yy,
                 x1x2yx1xx2_len, x1x2yx1my_len, x1x2yx1m_len,
                 x1x2yx1xx2yy_len):
        self.x1x2yx1xx2 = x1x2yx1xx2
        self.x1x2yx1xx2_len = x1x2yx1xx2_len

        self.x1x2yx1my = x1x2yx1my
        self.x1x2yx1my_len = x1x2yx1my_len
        self.x1x2yx1m_len = x1x2yx1m_len

        self.x1x2yx1xx2yy = x1x2yx1xx2yy
        self.x1x2yx1xx2yy_len = x1x2yx1xx2yy_len


    #x1x2_ids = tf.placeholder(tf.int32, shape=[None, None])
    #x1x2_len = tf.placeholder(tf.int32, shape=[None])
    #x1xx2_ids = tf.placeholder(tf.int32, shape=[None, None])
    #x1xx2_len = tf.placeholder(tf.int32, shape=[None])



#def _truncate_seqs(tokens_x1, tokens_x2, tokens_xx2, tokens_y, max_length):
#    while True:
#        total_length = len(tokens_x1) + len(tokens_x2) + len(tokens_xx2) + len(tokens_y)
#        if total_length <= max_length:
#            break
#        #tokens_x1.pop()
#        #tokens_x2.pop()
#        #tokens_xx2.pop()
#        tokens_y.pop()

def _truncate_seqs(x1, x2, xx2, y, max_length, encoder): # 裁剪掉超长的y or yy
    while True:
        ids = encoder.encode(x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2 + ' ' + y + ' ')
        if len(ids) <= max_length:
            break
        y_ = y.split()
        y = ' '.join(y_[:-1])
    return y

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal
    # percent of tokens from each, since if one sequence is very short then
    # each token that's truncated likely contains more information than a
    # longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def encode(tokenizer, max_seq_length, tokens_a, tokens_b=None):
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def process_single_example(mode, example, max_seq_length_x, max_seq_length_y, tokenizer):
    x1 = tokenizer.tokenize(example.x1)
    x2 = tokenizer.tokenize(example.x2)
    xx2 = tokenizer.tokenize(example.xx2)

    y1 = tokenizer.tokenize(example.y1)
    yy1_end1 = tokenizer.tokenize(example.yy1_end1)
    y2 = tokenizer.tokenize(example.y2)
    yy2_end1 = tokenizer.tokenize(example.yy2_end1)
    y3 = tokenizer.tokenize(example.y3)
    yy3_end1= tokenizer.tokenize(example.yy3_end1)

    input_ids_x1x2, input_mask_x1x2, segment_ids_x1x2 = encode(tokenizer,max_seq_length_x,x1,x2)
    input_ids_x1xx2, input_mask_x1xx2, segment_ids_x1xx2 = encode(tokenizer,max_seq_length_x,x1,xx2)
    input_ids_y1, input_mask_y1, segment_ids_y1 = encode(tokenizer,max_seq_length_y,y1)
    input_ids_y2, input_mask_y2, segment_ids_y2 = encode(tokenizer,max_seq_length_y,y2)
    input_ids_y3, input_mask_y3, segment_ids_y3 = encode(tokenizer,max_seq_length_y,y3)
    input_ids_yy1_end1, input_mask_yy1_end1, segment_ids_yy1_end1 = encode(tokenizer,max_seq_length_y,yy1_end1)
    input_ids_yy2_end1, input_mask_yy2_end1, segment_ids_yy2_end1 = encode(tokenizer,max_seq_length_y,yy2_end1)
    input_ids_yy3_end1, input_mask_yy3_end1, segment_ids_yy3_end1 = encode(tokenizer,max_seq_length_y,yy3_end1)


    if mode is 'dev_data' or mode is 'test_data':
        yy1_end2 = tokenizer.tokenize(example.yy1_end2)
        yy2_end2 = tokenizer.tokenize(example.yy2_end2)
        yy3_end2 = tokenizer.tokenize(example.yy3_end2)
        yy1_end3 = tokenizer.tokenize(example.yy1_end3)
        yy2_end3 = tokenizer.tokenize(example.yy2_end3)
        yy3_end3 = tokenizer.tokenize(example.yy3_end3)

        input_ids_yy1_end2, input_mask_yy1_end2, segment_ids_yy1_end2 = encode(tokenizer,max_seq_length_y,yy1_end2)
        input_ids_yy2_end2, input_mask_yy2_end2, segment_ids_yy2_end2 = encode(tokenizer,max_seq_length_y,yy2_end2)
        input_ids_yy3_end2, input_mask_yy3_end2, segment_ids_yy3_end2 = encode(tokenizer,max_seq_length_y,yy3_end2)

        input_ids_yy1_end3, input_mask_yy1_end3, segment_ids_yy1_end3 = encode(tokenizer,max_seq_length_y,yy1_end3)
        input_ids_yy2_end3, input_mask_yy2_end3, segment_ids_yy2_end3 = encode(tokenizer,max_seq_length_y,yy2_end3)
        input_ids_yy3_end3, input_mask_yy3_end3, segment_ids_yy3_end3 = encode(tokenizer,max_seq_length_y,yy3_end3)

    

        feature_x = {
            "input_ids_x1x2": input_ids_x1x2,
            "input_mask_x1x2": input_mask_x1x2,
            "segment_ids_x1x2": segment_ids_x1x2,
            "input_ids_x1xx2": input_ids_x1xx2,
            "input_mask_x1xx2": input_mask_x1xx2,
            "segment_ids_x1xx2": segment_ids_x1xx2,
        }
        feature_y = {
            "input_ids_y1": input_ids_y1,
            "input_mask_y1":input_mask_y1,
            "segment_ids_y1": segment_ids_y1,
            "input_ids_y2": input_ids_y2,
            "input_mask_y2":input_mask_y2,
            "segment_ids_y2": segment_ids_y2,
            "input_ids_y3": input_ids_y3,
            "input_mask_y3":input_mask_y3,
            "segment_ids_y3": segment_ids_y3,

            "input_ids_yy1_end1": input_ids_yy1_end1,
            "input_mask_yy1_end1":input_mask_yy1_end1,
            "segment_ids_yy1_end1": segment_ids_yy1_end1,
            "input_ids_yy2_end1": input_ids_yy2_end1,
            "input_mask_yy2_end1":input_mask_yy2_end1,
            "segment_ids_yy2_end1": segment_ids_yy2_end1,
            "input_ids_yy3_end1": input_ids_yy3_end1,
            "input_mask_yy3_end1":input_mask_yy3_end1,
            "segment_ids_yy3_end1": segment_ids_yy3_end1,

            "input_ids_yy1_end2": input_ids_yy1_end2,
            "input_mask_yy1_end2":input_mask_yy1_end2,
            "segment_ids_yy1_end2": segment_ids_yy1_end2,
            "input_ids_yy2_end2": input_ids_yy2_end2,
            "input_mask_yy2_end2":input_mask_yy2_end2,
            "segment_ids_yy2_end2": segment_ids_yy2_end2,
            "input_ids_yy3_end2": input_ids_yy3_end2,
            "input_mask_yy3_end2":input_mask_yy3_end2,
            "segment_ids_yy3_end2": segment_ids_yy3_end2,

            "input_ids_yy1_end3": input_ids_yy1_end3,
            "input_mask_yy1_end3":input_mask_yy1_end3,
            "segment_ids_yy1_end3": segment_ids_yy1_end3,
            "input_ids_yy2_end3": input_ids_yy2_end3,
            "input_mask_yy2_end3":input_mask_yy2_end3,
            "segment_ids_yy2_end3": segment_ids_yy2_end3,
            "input_ids_yy3_end3": input_ids_yy3_end3,
            "input_mask_yy3_end3":input_mask_yy3_end3,
            "segment_ids_yy3_end3": segment_ids_yy3_end3,
        }
    else:
        feature_x = {
            "input_ids_x1x2": input_ids_x1x2,
            "input_mask_x1x2": input_mask_x1x2,
            "segment_ids_x1x2": segment_ids_x1x2,
            "input_ids_x1xx2": input_ids_x1xx2,
            "input_mask_x1xx2": input_mask_x1xx2,
            "segment_ids_x1xx2": segment_ids_x1xx2,
        }
        feature_y = {
            "input_ids_y1": input_ids_y1,
            "input_mask_y1":input_mask_y1,
            "segment_ids_y1": segment_ids_y1,
            "input_ids_y2": input_ids_y2,
            "input_mask_y2":input_mask_y2,
            "segment_ids_y2": segment_ids_y2,
            "input_ids_y3": input_ids_y3,
            "input_mask_y3":input_mask_y3,
            "segment_ids_y3": segment_ids_y3,

            "input_ids_yy1_end1": input_ids_yy1_end1,
            "input_mask_yy1_end1":input_mask_yy1_end1,
            "segment_ids_yy1_end1": segment_ids_yy1_end1,
            "input_ids_yy2_end1": input_ids_yy2_end1,
            "input_mask_yy2_end1":input_mask_yy2_end1,
            "segment_ids_yy2_end1": segment_ids_yy2_end1,
            "input_ids_yy3_end1": input_ids_yy3_end1,
            "input_mask_yy3_end1":input_mask_yy3_end1,
            "segment_ids_yy3_end1": segment_ids_yy3_end1
        }

    return feature_x, feature_y



def read_raw_data_v2(path, mode):
    def _read_file(fn):
        with open(fn, 'r') as fin:
            lines = [line.strip() for line in fin]
        return lines

    def _get_fn(field):  
        return os.path.join(path, '%s_%s.txt' % (mode, field)) #train_y.txt # or '%s_%s.txt' % (mode, field) # 'TimeTravel.%s_%s.text' % (mode, field)

    all_x1 = _read_file(_get_fn('x1'))
    all_x2 = _read_file(_get_fn('x2'))
    all_xx2 = _read_file(_get_fn('xx2'))
    all_y = _read_file(_get_fn('y')) 
    
    all_y1 = _read_file(_get_fn('y1'))
    all_y2 = _read_file(_get_fn('y2'))
    all_y3 = _read_file(_get_fn('y3')) 
    
    #print('#examples: %d' % len(all_x1))

    if mode is 'dev_data' or mode is 'test_data':
        all_yy1_end1 = _read_file(_get_fn('yy1_end1'))
        all_yy2_end1 = _read_file(_get_fn('yy2_end1'))
        all_yy3_end1 = _read_file(_get_fn('yy3_end1'))
        
        all_yy1_end2 = _read_file(_get_fn('yy1_end2'))
        all_yy2_end2 = _read_file(_get_fn('yy2_end2'))
        all_yy3_end2 = _read_file(_get_fn('yy3_end2')) 
        
        all_yy1_end3 = _read_file(_get_fn('yy1_end3'))
        all_yy2_end3 = _read_file(_get_fn('yy2_end3'))
        all_yy3_end3 = _read_file(_get_fn('yy3_end3'))

        return [
            InputExample(
                x1=x1,
                x2=x2,
                xx2=xx2,
                y1=y1,
                y2=y2,
                y3=y3,
                y=y,
                yy1_end1=yy1_end1,
                yy2_end1=yy2_end1,
                yy3_end1=yy3_end1,
                yy1_end2=yy1_end2,
                yy2_end2=yy2_end2,
                yy3_end2=yy3_end2,
                yy1_end3=yy1_end3,
                yy2_end3=yy2_end3,
                yy3_end3=yy3_end3
                )
            for x1, x2, xx2, y1, y2, y3, y, yy1_end1, yy2_end1, yy3_end1, yy1_end2, yy2_end2, yy3_end2, yy1_end3, yy2_end3, yy3_end3 
            in zip(all_x1, all_x2, all_xx2, all_y1, all_y2, all_y3, all_y, all_yy1_end1, all_yy2_end1, 
            all_yy3_end1, all_yy1_end2, all_yy2_end2, all_yy3_end2, all_yy1_end3, all_yy2_end3, all_yy3_end3)
        ]
    else:
        all_yy1_end1 = _read_file(_get_fn('yy1'))
        all_yy2_end1 = _read_file(_get_fn('yy2'))
        all_yy3_end1 = _read_file(_get_fn('yy3'))
        return [
            InputExample(
                x1=x1,
                x2=x2,
                xx2=xx2,
                y1=y1,
                y2=y2,
                y3=y3,
                y=y,
                yy1_end1=yy1_end1,
                yy2_end1=yy2_end1,
                yy3_end1=yy3_end1
                )
                #yy=yy)
            for x1, x2, xx2, y1, y2, y3, y, yy1_end1, yy2_end1, yy3_end1 
            in zip(all_x1, all_x2, all_xx2, all_y1, all_y2, all_y3, all_y, all_yy1_end1, all_yy2_end1, all_yy3_end1)
        ]
        
    '''
    yy_fn = _get_fn('yy')
    if os.path.isfile(yy_fn):
        all_yy = _read_file(yy_fn)
    else:
        all_yy = [None] * len(all_x1)
    '''

    


def file_based_convert_examples_to_features_v2(mode,
        examples, max_seq_length_x, max_seq_length_y, tokenizer, output_file_x, output_file_y, verbose=False):
    """Converts a set of examples to a TFRecord file."""

    writer_x = tf.python_io.TFRecordWriter(output_file_x)
    writer_y = tf.python_io.TFRecordWriter(output_file_y)

    for (_, example) in enumerate(examples):

        fea_x, fea_y = process_single_example(mode,
            example, max_seq_length_x, max_seq_length_y, tokenizer)

        def _create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features_x = collections.OrderedDict()
        features_x["input_ids_x1x2"] = _create_int_feature(fea_x["input_ids_x1x2"])
        features_x["input_mask_x1x2"] = _create_int_feature(fea_x["input_mask_x1x2"])
        features_x["segment_ids_x1x2"] = _create_int_feature(fea_x["segment_ids_x1x2"])
        features_x["input_ids_x1xx2"] = _create_int_feature(fea_x["input_ids_x1xx2"])
        features_x["input_mask_x1xx2"] = _create_int_feature(fea_x["input_mask_x1xx2"])
        features_x["segment_ids_x1xx2"] = _create_int_feature(fea_x["segment_ids_x1xx2"])
        

        features_y = collections.OrderedDict()
        features_y["input_ids_y1"] = _create_int_feature(fea_y["input_ids_y1"])
        features_y["input_mask_y1"] = _create_int_feature(fea_y["input_mask_y1"])
        features_y["segment_ids_y1"] = _create_int_feature(fea_y["segment_ids_y1"])
        features_y["input_ids_y2"] = _create_int_feature(fea_y["input_ids_y2"])
        features_y["input_mask_y2"] = _create_int_feature(fea_y["input_mask_y2"])
        features_y["segment_ids_y2"] = _create_int_feature(fea_y["segment_ids_y2"])
        features_y["input_ids_y3"] = _create_int_feature(fea_y["input_ids_y3"])
        features_y["input_mask_y3"] = _create_int_feature(fea_y["input_mask_y3"])
        features_y["segment_ids_y3"] = _create_int_feature(fea_y["segment_ids_y3"])

        features_y["input_ids_yy1_end1"] = _create_int_feature(fea_y["input_ids_yy1_end1"])
        features_y["input_mask_yy1_end1"] = _create_int_feature(fea_y["input_mask_yy1_end1"])
        features_y["segment_ids_yy1_end1"] = _create_int_feature(fea_y["segment_ids_yy1_end1"])
        features_y["input_ids_yy2_end1"] = _create_int_feature(fea_y["input_ids_yy2_end1"])
        features_y["input_mask_yy2_end1"] = _create_int_feature(fea_y["input_mask_yy2_end1"])
        features_y["segment_ids_yy2_end1"] = _create_int_feature(fea_y["segment_ids_yy2_end1"])
        features_y["input_ids_yy3_end1"] = _create_int_feature(fea_y["input_ids_yy3_end1"])
        features_y["input_mask_yy3_end1"] = _create_int_feature(fea_y["input_mask_yy3_end1"])
        features_y["segment_ids_yy3_end1"] = _create_int_feature(fea_y["segment_ids_yy3_end1"])

        if mode is 'dev_data' or mode is 'test_data':
            features_y["input_ids_yy1_end2"] = _create_int_feature(fea_y["input_ids_yy1_end2"])
            features_y["input_mask_yy1_end2"] = _create_int_feature(fea_y["input_mask_yy1_end2"])
            features_y["segment_ids_yy1_end2"] = _create_int_feature(fea_y["segment_ids_yy1_end2"])
            features_y["input_ids_yy2_end2"] = _create_int_feature(fea_y["input_ids_yy2_end2"])
            features_y["input_mask_yy2_end2"] = _create_int_feature(fea_y["input_mask_yy2_end2"])
            features_y["segment_ids_yy2_end2"] = _create_int_feature(fea_y["segment_ids_yy2_end2"])
            features_y["input_ids_yy3_end2"] = _create_int_feature(fea_y["input_ids_yy3_end2"])
            features_y["input_mask_yy3_end2"] = _create_int_feature(fea_y["input_mask_yy3_end2"])
            features_y["segment_ids_yy3_end2"] = _create_int_feature(fea_y["segment_ids_yy3_end2"])

            features_y["input_ids_yy1_end3"] = _create_int_feature(fea_y["input_ids_yy1_end3"])
            features_y["input_mask_yy1_end3"] = _create_int_feature(fea_y["input_mask_yy1_end3"])
            features_y["segment_ids_yy1_end3"] = _create_int_feature(fea_y["segment_ids_yy1_end3"])
            features_y["input_ids_yy2_end3"] = _create_int_feature(fea_y["input_ids_yy2_end3"])
            features_y["input_mask_yy2_end3"] = _create_int_feature(fea_y["input_mask_yy2_end3"])
            features_y["segment_ids_yy2_end3"] = _create_int_feature(fea_y["segment_ids_yy2_end3"])
            features_y["input_ids_yy3_end3"] = _create_int_feature(fea_y["input_ids_yy3_end3"])
            features_y["input_mask_yy3_end3"] = _create_int_feature(fea_y["input_mask_yy3_end3"])
            features_y["segment_ids_yy3_end3"] = _create_int_feature(fea_y["segment_ids_yy3_end3"])


        tf_example_x = tf.train.Example(
            features=tf.train.Features(feature=features_x))
        writer_x.write(tf_example_x.SerializeToString())

        tf_example_y = tf.train.Example(
            features=tf.train.Features(feature=features_y))
        writer_y.write(tf_example_y.SerializeToString())


def prepare_TFRecord_data_v2(data_dir, max_seq_length_x, max_seq_length_y, tokenizer, output_dir):
    """
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord files in.
    """
    eval_examples = read_raw_data_v2(data_dir, mode='dev_data')
    print('##dev examples: %d' % len(eval_examples))
    eval_file_x = os.path.join(output_dir, "dev_data_x.tf_record")
    eval_file_y = os.path.join(output_dir, "dev_data_y.tf_record")
    file_based_convert_examples_to_features_v2('dev_data',
       eval_examples, max_seq_length_x, max_seq_length_y, tokenizer, eval_file_x, eval_file_y)

    test_examples = read_raw_data_v2(data_dir, mode='test_data')
    print('##test examples: %d' % len(test_examples))
    test_file_x = os.path.join(output_dir, "test_data_x.tf_record")
    test_file_y = os.path.join(output_dir, "test_data_y.tf_record")
    file_based_convert_examples_to_features_v2('test_data',
       test_examples, max_seq_length_x, max_seq_length_y, tokenizer, test_file_x, test_file_y, verbose=True)
  
    train_examples = read_raw_data_v2(data_dir, mode='train_supervised_large')#train_supervised_large or train data for mini
    print('##train examples: %d' % len(train_examples))
    train_file_x = os.path.join(output_dir, "{}_x.tf_record".format('train_supervised_large'))
    train_file_y = os.path.join(output_dir, "{}_y.tf_record".format('train_supervised_large'))
    file_based_convert_examples_to_features_v2('train_data',
        train_examples, max_seq_length_x, max_seq_length_y, tokenizer, train_file_x, train_file_y)



    '''
    train_examples = read_raw_data_v2(data_dir, mode='train_supervised_small')
    print('##train examples: %d' % len(train_examples))
    train_file = os.path.join(output_dir, "train_supervised_small.tf_record")
    file_based_convert_examples_to_features_v2(
        train_examples, max_seq_length, encoder, train_file)
    '''


'''
def process_single_example(example, max_seq_length_x, max_seq_length_y, tokenizer):
    x1 = example.x1
    x2 = example.x2
    xx2 = example.xx2
    y = example.y
    yy = example.yy
    y1 = example.y1
    yy1 = example.yy1
    y2 = example.y2
    yy2 = example.yy2
    y3 = example.y3
    yy3= example.yy3
    mask_text = 'Unknown .'
    special = tokenizer.encoder['<|endoftext|>']
    x1x2 = x1 + ' ' + x2
    x1x2_ids = encoder.encode(x1x2)
    x1xx2 = x1 + ' ' + xx2
    x1xx2_ids = encoder.encode(x1xx2)
    y1_ids = encoder.encode(y1)
    y2_ids = encoder.encode(y2)
    y3_ids = encoder.encode(y3)
    yy1_ids = encoder.encode(yy1)
    yy2_ids = encoder.encode(yy2)
    yy3_ids = encoder.encode(yy3)
    len_x1x2 = len(x1x2_ids)
    len_x1xx2 = len(x1xx2_ids)
    len_y1 = len(y1_ids)
    len_y2 = len(y2_ids)
    len_y3 = len(y3_ids)
    len_yy1 = len(yy1_ids)
    len_yy2 = len(yy2_ids)
    len_yy3 = len(yy3_ids)
    while len(x1x2_ids) < max_seq_length_x:
        x1x2_ids.append(special)
    while len(x1xx2_ids) < max_seq_length_x:
        x1xx2_ids.append(special)
    while len(y1_ids) < max_seq_length_y:
        y1_ids.append(special)
    while len(y2_ids) < max_seq_length_y:
        y2_ids.append(special)
    while len(y3_ids) < max_seq_length_y:
        y3_ids.append(special)
    while len(yy1_ids) < max_seq_length_y:
        yy1_ids.append(special)
    while len(yy2_ids) < max_seq_length_y:
        yy2_ids.append(special)
    while len(yy3_ids) < max_seq_length_y:
        yy3_ids.append(special)
    
    feature_x = {
        "x1x2_ids": x1x2_ids,
        "x1x2_len": len_x1x2,
        "x1xx2_ids": x1xx2_ids,
        "x1xx2_len": len_x1xx2
    }
    feature_y = {
        "y1_ids": y1_ids,
        "y1_len": len_y1,
        "y2_ids": y2_ids,
        "y2_len": len_y2,
        "y3_ids": y3_ids,
        "y3_len": len_y3,
        "yy1_ids": yy1_ids,
        "yy1_len": len_yy1,
        "yy2_ids": yy2_ids,
        "yy2_len": len_yy2,
        "yy3_ids": yy3_ids,
        "yy3_len": len_yy3
    }
    return feature_x, feature_y
'''