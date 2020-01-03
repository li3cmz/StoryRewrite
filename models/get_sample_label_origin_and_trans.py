'''
Use for split origin and trans dataset and then for ACCUEMD computation
'''
import os
import csv
from utils_data.utils import *


style_name_list = ['origin','trans']
version = 'GTAE_005_002' ###modify #v22_gcn_then_transformer-13-20
model_name = 'GTAE-adj-{}'.format(version)
file_name = 'val.20' ###modify
sample = './out_0903_main_{}/samples/{}'.format(version, file_name) 

sample_dir_name = '1000' ###modify 1000 or val.20
output_dir = 'Evaluation_model/ACCUEMD/pre-classifiers/data/yelp/'+model_name 
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
output_root = '{}/{}'.format(output_dir, sample_dir_name)
if os.path.exists(output_root) is False:
    os.mkdir(output_root)

text_label_dict = {}
line_list = []
if sample_dir_name is not '1000':

    val_text = './data/yelp/sentiment.dev.text'
    val_label = './data/yelp/sentiment.dev.labels'
    test_text = './data/yelp/sentiment.test.text'
    test_label = './data/yelp/sentiment.test.labels'

    # save val label and text
    with open(val_text) as textfile, open(val_label) as labelfile: 
        for line1 in labelfile:
            line2 = textfile.readline()
            if line2 is 'its worst - bad food , service worst atmosphere .':
                print(True)
            text_label_dict[line2] = int(line1[0])
            line_list.append(line2)
    # save test label and text  
    with open(test_text) as textfile, open(test_label) as labelfile: 
        for line1 in labelfile:
            line2 = textfile.readline()
            if line2 is 'its worst - bad food , service worst atmosphere .':
                print(True)
            text_label_dict[line2] = int(line1[0])
            line_list.append(line2)
else:
    random_500_0 = './data/yelp/random_500_0.text' ###modify
    random_500_1 = './data/yelp/random_500_1.text'
    cnt_500= 0
    with open(random_500_0) as textfile:
        line = textfile.readline()
        while line is not '':
            text_label_dict[line] = 0
            line_list.append(line)
            line = textfile.readline()
            cnt_500+=1
            #print(cnt_500)
            

    with open(random_500_1) as textfile:
        line = textfile.readline()
        while line is not '':
            text_label_dict[line] = 1
            line_list.append(line)
            line = textfile.readline()
            cnt_500+=1
            #print(cnt_500)
            

print(len(line_list))
for style_name in style_name_list:
    out_label = os.path.join(output_root, style_name + '_sample.labels') 
    out_text = os.path.join(output_root, style_name + '_sample.text')

    file_write_label = open(out_label, "w")
    file_write_text= open(out_text, "w")

    cnt_unk = 0
    line_number = 0
    with open(sample) as sample_text:
        line = sample_text.readline()
        
        line_number = 1
        while line is not '':
            print(line_number)
            if line not in line_list:
                #print(line)
                line = sample_text.readline()
                line = sample_text.readline()
                continue
            #''' use for val.20 val.12 because too many unk we can't process it one by one before 0813
            if line.count('<UNK>') != 0:
                tmp_cnt = line.count('<UNK>') 
                trans_line = sample_text.readline()
                if trans_line.count('<UNK>') != 0:
                    cnt_unk+=trans_line.count('<UNK>')
                line = sample_text.readline()
                line_number += 2
                cnt_unk+=tmp_cnt
                
                continue
            #'''
            if line is '':
                break

            label = text_label_dict[line]
            text = line

            if style_name is 'origin':
                file_write_label.writelines(str(label)+'\n') # ori 
                file_write_text.writelines(text) # ori 
            elif style_name is 'trans':
                file_write_label.writelines(str(1-label)+'\n') # trans  
                  
            trans_text = sample_text.readline()
            #'''use for val.20 val.12 because too many unk we can't process it one by one before 0813
            if trans_text.count('<UNK>') != 0:
                    cnt_unk+=trans_text.count('<UNK>') 
            #'''
            if style_name is 'trans':
                file_write_text.writelines(trans_text) # trans
            line = sample_text.readline()

            line_number += 2