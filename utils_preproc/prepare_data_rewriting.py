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
"""Preprocesses raw data and produces TFRecord files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import texar as tx

import processor
import data_utils_rewriting_2 as data_utils
import tokenization

# pylint: disable=invalid-name, too-many-locals, too-many-statements
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(3) ### modify


flags = tf.flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_dir", '../data/TimeTravel/mini', #mini or pre
    "The directory of raw data, wherein data files must be named as "
    "'train.txt', 'dev.txt', or 'test.txt'.") ###modify
flags.DEFINE_string(
    "vocab_file", '../bert_pretrained_models/bert_pretrained_models/cased_L-12_H-768_A-12/vocab.txt', ###modify
    "The one-wordpiece-per-line vocabary file directory.")
flags.DEFINE_integer(
    "max_seq_length_x", 128, #2sentence
    "The maxium length of sequence of x, longer sequence will be trimmed.")
flags.DEFINE_integer(
    "max_seq_length_y", 128,
    "The maxium length of sequence of y, longer sequence will be trimmed.")
flags.DEFINE_string(
    "tfrecord_output_dir", '../data/TimeTravel/bert3_case/remove',
    "The output directory where the TFRecord files will be generated. "
    "By default it is set to be the same as `--data_dir`.") ###modify bert2/mini or bert2
flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


tf.logging.set_verbosity(tf.logging.INFO)


def prepare_data():
    """
    Builds the model and runs.
    """
    data_dir = FLAGS.data_dir
    if FLAGS.tfrecord_output_dir is None:
        tfrecord_output_dir = data_dir
    else:
        tfrecord_output_dir = FLAGS.tfrecord_output_dir
    tx.utils.maybe_create_dir(tfrecord_output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file,
        do_lower_case=FLAGS.do_lower_case)

    # Produces TFRecord files
    data_utils.prepare_TFRecord_data_v2(
        data_dir=data_dir,
        max_seq_length_x=FLAGS.max_seq_length_x,
        max_seq_length_y=FLAGS.max_seq_length_y,
        tokenizer=tokenizer,
        output_dir=tfrecord_output_dir)


def main():
    """Data preparation.
    """
    prepare_data()


if __name__ == "__main__":
    main()