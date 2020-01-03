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
import tokenization
# pylint: disable=invalid-name, too-many-arguments

class InputExample(object):

    def __init__(self, x1, x2, xx2, y1, y2, y3, y, yy1, yy2, yy3, y1_gpt, y2_gpt, y3_gpt):
        self.x1 = x1
        self.x2 = x2
        self.xx2 = xx2
        self.y = y
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

        self.yy1=yy1
        self.yy2=yy2
        self.yy3=yy3

        self.y1_gpt=y1_gpt
        self.y2_gpt=y2_gpt
        self.y3_gpt=y3_gpt
        



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

    return input_ids, input_mask, segment_ids, tokens

def process_single_example(mode, example, max_seq_length_x, max_seq_length_y, tokenizer):
    x1 = example.x1
    x2 = example.x2
    xx2 = example.xx2
    y = example.y

    yy1 = example.yy1
    yy2 = example.yy2
    yy3 = example.yy3

    y1_gpt = example.y1_gpt
    y2_gpt = example.y2_gpt
    y3_gpt = example.y3_gpt

    
    x1x2ysx1xx2 = tokenizer.tokenize(x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2)
    x1x2ysx1xx2yy = tokenizer.tokenize(x1 + ' ' + x2 + ' ' + y + ' | ' + x1 + ' ' + xx2 + ' ' + yy1 + ' ' + yy2 + ' ' + yy3)
    x1x2 = tokenizer.tokenize(x1 + ' ' + x2)
    x1xx2 = tokenizer.tokenize(x1 + ' ' + xx2)
    y1 = tokenizer.tokenize(example.y1)
    yy1 = tokenizer.tokenize(example.yy1)
    y2 = tokenizer.tokenize(example.y2)
    yy2 = tokenizer.tokenize(example.yy2)
    y3 = tokenizer.tokenize(example.y3)
    yy3= tokenizer.tokenize(example.yy3)
    y1_gpt = tokenizer.tokenize(y1_gpt)
    y2_gpt = tokenizer.tokenize(y2_gpt)
    y3_gpt = tokenizer.tokenize(y3_gpt)

    input_ids_x1x2ysx1xx2, input_mask_x1x2ysx1xx2, segment_ids_x1x2ysx1xx2, _ = encode(tokenizer,max_seq_length_x, x1x2ysx1xx2)
    input_ids_x1x2ysx1xx2yy, input_mask_x1x2ysx1xx2yy, segment_ids_x1x2ysx1xx2yy, _ = encode(tokenizer,max_seq_length_x, x1x2ysx1xx2yy)

    input_ids_x1x2, input_mask_x1x2, segment_ids_x1x2, tokens = encode(tokenizer,max_seq_length_x,x1x2)
    input_ids_x1xx2, input_mask_x1xx2, segment_ids_x1xx2, _ = encode(tokenizer,max_seq_length_x,x1xx2)
    input_ids_y1, input_mask_y1, segment_ids_y1, _ = encode(tokenizer,max_seq_length_y,y1)
    input_ids_y2, input_mask_y2, segment_ids_y2, _ = encode(tokenizer,max_seq_length_y,y2)
    input_ids_y3, input_mask_y3, segment_ids_y3, _ = encode(tokenizer,max_seq_length_y,y3)
    input_ids_yy1, input_mask_yy1, segment_ids_yy1, _ = encode(tokenizer,max_seq_length_y,yy1)
    input_ids_yy2, input_mask_yy2, segment_ids_yy2, _ = encode(tokenizer,max_seq_length_y,yy2)
    input_ids_yy3, input_mask_yy3, segment_ids_yy3, _ = encode(tokenizer,max_seq_length_y,yy3)

    input_ids_y1_gpt, input_mask_y1_gpt, segment_ids_y1_gpt, _ = encode(tokenizer,max_seq_length_y,y1_gpt)
    input_ids_y2_gpt, input_mask_y2_gpt, segment_ids_y2_gpt, _ = encode(tokenizer,max_seq_length_y,y2_gpt)
    input_ids_y3_gpt, input_mask_y3_gpt, segment_ids_y3_gpt, _ = encode(tokenizer,max_seq_length_y,y3_gpt)

    '''
    tf.logging.info("*** Example ***")
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    '''
        
    feature_x = {
        "input_ids_x1x2ysx1xx2": input_ids_x1x2ysx1xx2,
        "input_mask_x1x2ysx1xx2": input_mask_x1x2ysx1xx2,
        "segment_ids_x1x2ysx1xx2": segment_ids_x1x2ysx1xx2,
        "input_ids_x1x2ysx1xx2yy": input_ids_x1x2ysx1xx2yy,
        "input_mask_x1x2ysx1xx2yy": input_mask_x1x2ysx1xx2yy,
        "segment_ids_x1x2ysx1xx2yy": segment_ids_x1x2ysx1xx2yy,
        

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

        "input_ids_yy1": input_ids_yy1,
        "input_mask_yy1":input_mask_yy1,
        "segment_ids_yy1": segment_ids_yy1,
        "input_ids_yy2": input_ids_yy2,
        "input_mask_yy2":input_mask_yy2,
        "segment_ids_yy2": segment_ids_yy2,
        "input_ids_yy3": input_ids_yy3,
        "input_mask_yy3":input_mask_yy3,
        "segment_ids_yy3": segment_ids_yy3,

        "input_ids_y1_gpt": input_ids_y1_gpt,
        "input_mask_y1_gpt":input_mask_y1_gpt,
        "segment_ids_y1_gpt": segment_ids_y1_gpt,
        "input_ids_y2_gpt": input_ids_y2_gpt,
        "input_mask_y2_gpt":input_mask_y2_gpt,
        "segment_ids_y2_gpt": segment_ids_y2_gpt,
        "input_ids_y3_gpt": input_ids_y3_gpt,
        "input_mask_y3_gpt":input_mask_y3_gpt,
        "segment_ids_y3_gpt": segment_ids_y3_gpt,
    }

    return feature_x, feature_y



def read_raw_data_v2(path, mode):
    def _read_file(fn):
        with open(fn, 'r') as fin:
            lines = [line.strip() for line in fin]
        return lines

    def _get_fn(path, field): 
       
        return os.path.join(path, 'TimeTravel.%s_%s.txt' % (mode, field)) #train_y.txt # or '%s_%s.txt' % (mode, field) # 'TimeTravel.%s_%s.text' % (mode, field) 
        ###modify for big:'%s_%s.txt' % (mode, field)  for mini: 'TimeTravel.%s_%s.txt' % (mode, field)

    all_x1 = _read_file(_get_fn(path, 'x1'))
    all_x2 = _read_file(_get_fn(path, 'x2'))
    all_xx2 = _read_file(_get_fn(path, 'xx2'))
    all_y = _read_file(_get_fn(path, 'y'))
    
    all_y1 = _read_file(_get_fn(path, 'y1'))
    all_y2 = _read_file(_get_fn(path, 'y2'))
    all_y3 = _read_file(_get_fn(path, 'y3'))

    all_y1_gpt = _read_file(_get_fn(path + '/../gpt-2-refine/pre/mini', 'gpt2_y1'))  ###'pre/mini'or 'pre'
    all_y2_gpt = _read_file(_get_fn(path + '/../gpt-2-refine/pre/mini', 'gpt2_y2')) 
    all_y3_gpt = _read_file(_get_fn(path + '/../gpt-2-refine/pre/mini', 'gpt2_y3'))
    
    #print('#examples: %d' % len(all_x1))

    if mode is 'dev_data' or mode is 'test_data':
        all_yy1 = _read_file(_get_fn(path, 'yy1_end1'))
        all_yy2 = _read_file(_get_fn(path, 'yy2_end1'))
        all_yy3 = _read_file(_get_fn(path, 'yy3_end1'))
    else:
        all_yy1 = _read_file(_get_fn(path, 'yy1'))
        all_yy2 = _read_file(_get_fn(path, 'yy2'))
        all_yy3 = _read_file(_get_fn(path, 'yy3'))

    return [
        InputExample(
            x1=x1,
            x2=x2,
            xx2=xx2,
            y1=y1,
            y2=y2,
            y3=y3,
            y=y,
            yy1=yy1,
            yy2=yy2,
            yy3=yy3,
            y1_gpt=y1_gpt,
            y2_gpt=y2_gpt,
            y3_gpt=y3_gpt
            )
        for x1, x2, xx2, y1, y2, y3, y, yy1, yy2, yy3, y1_gpt, y2_gpt, y3_gpt
        in zip(all_x1, all_x2, all_xx2, all_y1, all_y2, all_y3, all_y, all_yy1, all_yy2, 
        all_yy3, all_y1_gpt, all_y2_gpt, all_y3_gpt)
    ]
        
    '''
    yy_fn = _get_fn('yy')
    if os.path.isfile(yy_fn):
        all_yy = _read_file(yy_fn)
    else:
        all_yy = [None] * len(all_x1)
    '''

    


def file_based_convert_examples_to_features_v2(mode,
        examples, max_seq_length_x, max_seq_length_y, tokenizer, output_file, verbose=False):
    """Converts a set of examples to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (_, example) in enumerate(examples):

        fea_x, fea_y = process_single_example(mode,
            example, max_seq_length_x, max_seq_length_y, tokenizer)

        def _create_int_feature(values):
            return tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids_x1x2ysx1xx2"] = _create_int_feature(fea_x["input_ids_x1x2ysx1xx2"])
        features["input_mask_x1x2ysx1xx2"] = _create_int_feature(fea_x["input_mask_x1x2ysx1xx2"])
        features["segment_ids_x1x2ysx1xx2"] = _create_int_feature(fea_x["segment_ids_x1x2ysx1xx2"])
        features["input_ids_x1x2ysx1xx2yy"] = _create_int_feature(fea_x["input_ids_x1x2ysx1xx2yy"])
        features["input_mask_x1x2ysx1xx2yy"] = _create_int_feature(fea_x["input_mask_x1x2ysx1xx2yy"])
        features["segment_ids_x1x2ysx1xx2yy"] = _create_int_feature(fea_x["segment_ids_x1x2ysx1xx2yy"])
        
        features["input_ids_x1x2"] = _create_int_feature(fea_x["input_ids_x1x2"])
        features["input_mask_x1x2"] = _create_int_feature(fea_x["input_mask_x1x2"])
        features["segment_ids_x1x2"] = _create_int_feature(fea_x["segment_ids_x1x2"])
        features["input_ids_x1xx2"] = _create_int_feature(fea_x["input_ids_x1xx2"])
        features["input_mask_x1xx2"] = _create_int_feature(fea_x["input_mask_x1xx2"])
        features["segment_ids_x1xx2"] = _create_int_feature(fea_x["segment_ids_x1xx2"])
        
        features["input_ids_y1"] = _create_int_feature(fea_y["input_ids_y1"])
        features["input_mask_y1"] = _create_int_feature(fea_y["input_mask_y1"])
        features["segment_ids_y1"] = _create_int_feature(fea_y["segment_ids_y1"])
        features["input_ids_y2"] = _create_int_feature(fea_y["input_ids_y2"])
        features["input_mask_y2"] = _create_int_feature(fea_y["input_mask_y2"])
        features["segment_ids_y2"] = _create_int_feature(fea_y["segment_ids_y2"])
        features["input_ids_y3"] = _create_int_feature(fea_y["input_ids_y3"])
        features["input_mask_y3"] = _create_int_feature(fea_y["input_mask_y3"])
        features["segment_ids_y3"] = _create_int_feature(fea_y["segment_ids_y3"])

        features["input_ids_yy1"] = _create_int_feature(fea_y["input_ids_yy1"])
        features["input_mask_yy1"] = _create_int_feature(fea_y["input_mask_yy1"])
        features["segment_ids_yy1"] = _create_int_feature(fea_y["segment_ids_yy1"])
        features["input_ids_yy2"] = _create_int_feature(fea_y["input_ids_yy2"])
        features["input_mask_yy2"] = _create_int_feature(fea_y["input_mask_yy2"])
        features["segment_ids_yy2"] = _create_int_feature(fea_y["segment_ids_yy2"])
        features["input_ids_yy3"] = _create_int_feature(fea_y["input_ids_yy3"])
        features["input_mask_yy3"] = _create_int_feature(fea_y["input_mask_yy3"])
        features["segment_ids_yy3"] = _create_int_feature(fea_y["segment_ids_yy3"])

        features["input_ids_y1_gpt"] = _create_int_feature(fea_y["input_ids_y1_gpt"])
        features["input_mask_y1_gpt"] = _create_int_feature(fea_y["input_mask_y1_gpt"])
        features["segment_ids_y1_gpt"] = _create_int_feature(fea_y["segment_ids_y1_gpt"])
        features["input_ids_y2_gpt"] = _create_int_feature(fea_y["input_ids_y2_gpt"])
        features["input_mask_y2_gpt"] = _create_int_feature(fea_y["input_mask_y2_gpt"])
        features["segment_ids_y2_gpt"] = _create_int_feature(fea_y["segment_ids_y2_gpt"])
        features["input_ids_y3_gpt"] = _create_int_feature(fea_y["input_ids_y3_gpt"])
        features["input_mask_y3_gpt"] = _create_int_feature(fea_y["input_mask_y3_gpt"])
        features["segment_ids_y3_gpt"] = _create_int_feature(fea_y["segment_ids_y3_gpt"])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

       

def prepare_TFRecord_data_v2(data_dir, max_seq_length_x, max_seq_length_y, tokenizer, output_dir):
    """
    Args:
        data_dir: The input data directory.
        max_seq_length: Max sequence length.
        output_dir: The directory to save the TFRecord files in.
    """
    eval_examples = read_raw_data_v2(data_dir, mode='dev_data')
    print('##dev examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "dev_data.tf_record")
    file_based_convert_examples_to_features_v2('dev_data',
       eval_examples, max_seq_length_x, max_seq_length_y, tokenizer, eval_file)

    test_examples = read_raw_data_v2(data_dir, mode='test_data')
    print('##test examples: %d' % len(test_examples))
    test_file = os.path.join(output_dir, "test_data.tf_record")
    file_based_convert_examples_to_features_v2('test_data',
       test_examples, max_seq_length_x, max_seq_length_y, tokenizer, test_file, verbose=True)
  
    train_examples = read_raw_data_v2(data_dir, mode='train_data')#train_supervised_large for big or train data for mini  ###modify
    print('##train examples: %d' % len(train_examples))
    train_file = os.path.join(output_dir, "{}.tf_record".format('train_supervised_large'))
    file_based_convert_examples_to_features_v2('train_data',
        train_examples, max_seq_length_x, max_seq_length_y, tokenizer, train_file)



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