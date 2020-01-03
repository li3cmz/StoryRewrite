"""
Use for out_put_dir/sample/val.20.random_1000
"""
import os
import csv
from utils_data.utils import *


version = 'GTAE_005_002' ###modify
file_name = 'val.20' ###modify
date = 'out_0903' ###modify
model_name = 'GTAE-adj-{}'.format(version)
sample = './{}_main_{}/samples/{}'.format(date, version, file_name)

saved_text = './data/yelp/random_1000.text'

saved_sample_list = []
tag_bool = []
repeat_sample_in_random_list = []
with open(saved_text) as textfile: 
    line = textfile.readline()
    line_number = 1
    while line is not '':
        saved_sample_list.append(line)
        tag_bool.append(False)
        line = textfile.readline()
        line_number+=1

output_root = './{}_main_{}/samples'.format(date, version)
out_text = os.path.join(output_root,  file_name+'.random_1000')
file_write_text= open(out_text, "w")
unk_cnt = 0
cnt_repeat_sample = 0
repeat_sample_list = []
ok_str= []
with open(sample) as textfile:
    line = textfile.readline()
    line_number = 0
    while line is not '':
        idx = None
        maybe_repeat_str = line
        if line.count('<UNK>') != 0: #有UNK的先不要拉，因为后面有UNK没法弄
            unk_cnt +=line.count('<UNK>')
        if line in saved_sample_list:
            idx = saved_sample_list.index(line)
        if line in saved_sample_list and tag_bool[idx] is False:#在1000 sample里面，且还没有被写入val.20.random_1000里面。考虑丢失的4个sample是不是有UNK导致，第一个条件判断错误，导致sample变少，我们无法对有unk的在random_1000里面的进行判断，后面就不会出现这种问题拉
            tag_bool[idx] = True
            file_write_text.writelines(line)
            ok_str.append(line)
            trans_line = textfile.readline()
            file_write_text.writelines(trans_line)
        else:
            if idx is not None and tag_bool[idx] is True:#首先要在1000个sample里面，其次要已经是重复状态了
                if maybe_repeat_str not in repeat_sample_list:
                    repeat_sample_list.append(maybe_repeat_str)
                print(line, len(repeat_sample_list))
            trans_line = textfile.readline()

        if trans_line.count('<UNK>') != 0: #有UNK的先不要拉，因为后面有UNK没法弄
            unk_cnt +=trans_line.count('<UNK>')
        line_number+=2

        line = textfile.readline()
        #print(line_number, unk_cnt)

### 检查一下是漏掉了哪个样本，可以看到是UNK导致的
cnt_lou = 0
for strr in saved_sample_list:
    if strr not in ok_str:
        cnt_lou +=1
        print(strr, cnt_lou)
