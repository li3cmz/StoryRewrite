import csv
from utils_data.utils import *

train_text = './data/yelp/sentiment.train.text'
train_label = './data/yelp/sentiment.train.labels'
val_text = './data/yelp/sentiment.dev.text'
val_label = './data/yelp/sentiment.dev.labels'
test_text = './data/yelp/sentiment.test.text'
test_label = './data/yelp/sentiment.test.labels'

train_out_1 = "./data/yelp/yelp.train.1"
train_out_0 = "./data/yelp/yelp.train.1"
val_test_out_1 = "./data/yelp/yelp.test.1"
val_test_out_0 = "./data/yelp/yelp.test.1"



file_write_obj_0 = open(val_test_out_0, "w")

with open(test_label) as labelfile, open(test_text) as textfile: 
    for line1 in labelfile:
        line2 = textfile.readline()
        if int(line1[0]) is 1:
            file_write_obj_0.writelines(line2)
        #else:
        #    file_write_obj_1.writelines(line2)
            


'''
train_text_csv = csv.reader(open(train_text,'r'))
train_label_csv = csv.reader(open(train_label,'r'))
val_text_csv = csv.reader(open(val_text,'r'))
val_label_csv = csv.reader(open(val_label,'r'))
test_text_csv = csv.reader(open(test_text,'r'))
test_label_csv = csv.reader(open(test_label,'r'))


for line in train_text_csv:
    
'''
