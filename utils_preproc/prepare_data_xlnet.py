import tensorflow as tf
import sentencepiece as spm
import os
from xlnet_prepro_utils import *
import collections
import csv

def _get_spm_basename(spiece_model_file):
      spm_basename = os.path.basename(spiece_model_file)
      return spm_basename


class InputExample(object):
    
    def __init__(self, x1, x2, xx2, y1, y2, y3, y, yy1, yy2, yy3):
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
        

def read_raw_data_v2(path, mode):
    def _read_file(fn):
        with open(fn, 'r') as fin:
            lines = [line.strip() for line in fin]
        return lines

    def _get_fn(field):  
        return os.path.join(path, 'TimeTravel.%s_%s.txt' % (mode, field)) #train_y.txt # or '%s_%s.txt' % (mode, field) # 'TimeTravel.%s_%s.text' % (mode, field) 
        ###modify for big:'%s_%s.txt' % (mode, field)  for mini: 'TimeTravel.%s_%s.txt' % (mode, field)

    all_x1 = _read_file(_get_fn('x1'))
    all_x2 = _read_file(_get_fn('x2'))
    all_xx2 = _read_file(_get_fn('xx2'))
    all_y = _read_file(_get_fn('y'))
    
    all_y1 = _read_file(_get_fn('y1'))
    all_y2 = _read_file(_get_fn('y2'))
    all_y3 = _read_file(_get_fn('y3')) 
    
    #print('#examples: %d' % len(all_x1))

    if mode is 'dev_data' or mode is 'test_data':
        all_yy1 = _read_file(_get_fn('yy1_end1'))
        all_yy2 = _read_file(_get_fn('yy2_end1'))
        all_yy3 = _read_file(_get_fn('yy3_end1'))
    else:
        all_yy1 = _read_file(_get_fn('yy1'))
        all_yy2 = _read_file(_get_fn('yy2'))
        all_yy3 = _read_file(_get_fn('yy3'))

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
            yy3=yy3
            )
        for x1, x2, xx2, y1, y2, y3, y, yy1, yy2, yy3 
        in zip(all_x1, all_x2, all_xx2, all_y1, all_y2, all_y3, all_y, all_yy1, all_yy2, 
        all_yy3)
    ]
    
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


def encode(sp_model, max_seq_length, lower, texta, textb=None):
    CLS_ID = 3
    SEP_ID = 4
    '''
    SEG_ID_P = 0
    SEG_ID_Q = 1
    SEG_ID_CLS = 2
    SEG_ID_PAD = 3
    '''
    texta_tokens = encode_ids(sp_model, preprocess_text(texta, lower=lower)) #已经是id形式了
    if textb:
        textb_tokens = encode_ids(sp_model, preprocess_text(textb, lower=lower)) #已经是id形式了
        _truncate_seq_pair(texta_tokens, textb_tokens, max_seq_length - 3)
    else:
        if len(texta_tokens) > max_seq_length - 2:
                texta_tokens = texta_tokens[0:(max_seq_length - 2)]
        

    tokens = []
    tokens.append(CLS_ID)
    for token in texta_tokens:
        tokens.append(token)
    tokens.append(SEP_ID)

    if textb:
        for token in textb_tokens:
            tokens.append(token)
        tokens.append(SEP_ID)


    input_ids = tokens
    input_mask = [0] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(1)
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    print(input_ids, input_mask)
    return input_ids, input_mask


def process_single_example(mode, sp_model, example, max_seq_length, lower):
    x1 = example.x1
    x2 = example.x2
    xx2 = example.xx2
    y = example.y
    yy1 = example.yy1
    yy2 = example.yy2
    yy3 = example.yy3

    x1x2y = x1 + ' ' + x2 + ' ' + y
    x1xx2 = x1 + ' ' + xx2
    x1xx2yy = x1 + ' ' + xx2 + ' ' + yy1 + ' ' + yy2 + ' ' + yy3
    x1x2ysx1xx2_input_ids, x1x2ysx1xx2_input_mask = encode(sp_model, max_seq_length, lower, x1x2y, x1xx2)
    x1x2ysx1xx2yy_input_ids, x1x2ysx1xx2yy_input_mask = encode(sp_model, max_seq_length, lower, x1x2y, x1xx2yy)
    
    yy = yy1 + ' ' + yy2 + ' ' + yy3
    x1x2 = x1 + ' ' + x2
    y = example.y
    y1 = example.y1
    y2 = example.y2
    y3 = example.y3
    yy1 = example.yy1
    yy2 = example.yy2
    yy3 = example.yy3

    y_input_ids, y_input_mask = encode(sp_model, max_seq_length, lower, y)
    y1_input_ids, y1_input_mask = encode(sp_model, max_seq_length, lower, y1)
    y2_input_ids, y2_input_mask = encode(sp_model, max_seq_length, lower, y2)
    y3_input_ids, y3_input_mask = encode(sp_model, max_seq_length, lower, y3)

    yy_input_ids, yy_input_mask = encode(sp_model, max_seq_length, lower, yy)
    yy1_input_ids, yy1_input_mask = encode(sp_model, max_seq_length, lower, yy1)
    yy2_input_ids, yy2_input_mask = encode(sp_model, max_seq_length, lower, yy2)
    yy3_input_ids, yy3_input_mask = encode(sp_model, max_seq_length, lower, yy3)

    x1x2_input_ids, x1x2_input_mask = encode(sp_model, max_seq_length, lower, x1x2)
    x1xx2_input_ids, x1xx2_input_mask = encode(sp_model, max_seq_length, lower, x1xx2)

        
    feature_x = {
        "input_ids_x1x2ysx1xx2": x1x2ysx1xx2_input_ids,
        "input_mask_x1x2ysx1xx2": x1x2ysx1xx2_input_mask,
        "input_ids_x1x2ysx1xx2yy": x1x2ysx1xx2yy_input_ids,
        "input_mask_x1x2ysx1xx2yy": x1x2ysx1xx2yy_input_mask,

        "input_ids_x1x2": x1x2_input_ids,
        "input_mask_x1x2": x1x2_input_mask,
        "input_ids_x1xx2": x1xx2_input_ids,
        "input_mask_x1xx2": x1xx2_input_mask,

        "input_ids_yy": yy_input_ids,
        "input_mask_yy": yy_input_mask,
        "input_ids_y": y_input_ids,
        "input_mask_y": y_input_mask,
    }
    feature_y = {
        "input_ids_y1": y1_input_ids,
        "input_mask_y1":y1_input_mask,
        "input_ids_y2": y2_input_ids,
        "input_mask_y2":y2_input_mask,
        "input_ids_y3": y3_input_ids,
        "input_mask_y3":y3_input_mask,

        "input_ids_yy1": yy1_input_ids,
        "input_mask_yy1":yy1_input_mask,
        "input_ids_yy2": yy2_input_ids,
        "input_mask_yy2":yy2_input_mask,
        "input_ids_yy3": yy3_input_ids,
        "input_mask_yy3": yy3_input_mask,
    }

    return feature_x, feature_y

def file_based_convert_examples_to_features_v2(mode, sp_model,
        examples, max_seq_length, output_file, lower, verbose=False):
      """Converts a set of examples to a TFRecord file."""

      writer = tf.python_io.TFRecordWriter(output_file)

      for (_, example) in enumerate(examples):

            fea_x, fea_y = process_single_example(mode, sp_model,
            example, max_seq_length, lower)

            def _create_int_feature(values):
                  return tf.train.Feature(
                        int64_list=tf.train.Int64List(value=list(values)))

            features = collections.OrderedDict()
            features["input_ids_x1x2ysx1xx2"] = _create_int_feature(fea_x["input_ids_x1x2ysx1xx2"])
            features["input_mask_x1x2ysx1xx2"] = _create_int_feature(fea_x["input_mask_x1x2ysx1xx2"])
            features["input_ids_x1x2ysx1xx2yy"] = _create_int_feature(fea_x["input_ids_x1x2ysx1xx2yy"])
            features["input_mask_x1x2ysx1xx2yy"] = _create_int_feature(fea_x["input_mask_x1x2ysx1xx2yy"])

            features["input_ids_x1x2"] = _create_int_feature(fea_x["input_ids_x1x2"])
            features["input_mask_x1x2"] = _create_int_feature(fea_x["input_mask_x1x2"])
            features["input_ids_x1xx2"] = _create_int_feature(fea_x["input_ids_x1xx2"])
            features["input_mask_x1xx2"] = _create_int_feature(fea_x["input_mask_x1xx2"])

            features["input_ids_yy"] = _create_int_feature(fea_x["input_ids_yy"])
            features["input_mask_yy"] = _create_int_feature(fea_x["input_mask_yy"])
            features["input_ids_y"] = _create_int_feature(fea_x["input_ids_y"])
            features["input_mask_y"] = _create_int_feature(fea_x["input_mask_y"])

            features["input_ids_y1"] = _create_int_feature(fea_y["input_ids_y1"])
            features["input_mask_y1"] = _create_int_feature(fea_y["input_mask_y1"])
            features["input_ids_y2"] = _create_int_feature(fea_y["input_ids_y2"])
            features["input_mask_y2"] = _create_int_feature(fea_y["input_mask_y2"])
            features["input_ids_y3"] = _create_int_feature(fea_y["input_ids_y3"])
            features["input_mask_y3"] = _create_int_feature(fea_y["input_mask_y3"])

            features["input_ids_yy1"] = _create_int_feature(fea_y["input_ids_yy1"])
            features["input_mask_yy1"] = _create_int_feature(fea_y["input_mask_yy1"])
            features["input_ids_yy2"] = _create_int_feature(fea_y["input_ids_yy2"])
            features["input_mask_yy2"] = _create_int_feature(fea_y["input_mask_yy2"])
            features["input_ids_yy3"] = _create_int_feature(fea_y["input_ids_yy3"])
            features["input_mask_yy3"] = _create_int_feature(fea_y["input_mask_yy3"])


            tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

       

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    output_dir = "../data/TimeTravel/xlnet/mini" ###modify
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    sp_model = spm.SentencePieceProcessor()
    spiece_model_file = '../xlnet_cased_L-12_H-768_A-12/spiece.model' ###modify
    sp_model.Load(spiece_model_file)
    spm_basename = _get_spm_basename(spiece_model_file)
    data_dir = '../data/TimeTravel/mini' ###modify
    max_seq_length = 128
    lower = False

    eval_examples = read_raw_data_v2(data_dir, mode='dev_data')
    print('##dev examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "dev_data.tf_record")
    file_based_convert_examples_to_features_v2('dev_data', sp_model,
        eval_examples, max_seq_length, eval_file, lower)

    eval_examples = read_raw_data_v2(data_dir, mode='test_data')
    print('##test examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "test_data.tf_record")
    file_based_convert_examples_to_features_v2('test_data', sp_model,
        eval_examples, max_seq_length, eval_file, lower)


    eval_examples = read_raw_data_v2(data_dir, mode='train_data') #'train_data #modify
    print('##train examples: %d' % len(eval_examples))
    eval_file = os.path.join(output_dir, "train_supervised_large.tf_record")
    file_based_convert_examples_to_features_v2('train_data', sp_model,
        eval_examples, max_seq_length, eval_file, lower)



