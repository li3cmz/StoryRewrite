import os
import csv

vocab_train = './data/yelp/vocab_yelp_train'
vocab_test = './data/yelp/vocab_yelp_test'
vocab_dev = './data/yelp/vocab_yelp_dev'


vocab_list = []
cnt_add = 0 # 统计val和test合起来比train多了多少,就是自己原来忽略了多少
for i in range(3):
    if i == 0:
        vocab_file = vocab_train
    elif i == 1:
        vocab_file = vocab_dev
    elif i == 2:
        vocab_file = vocab_test
    with open(vocab_file) as vocab_iter: 
        line = vocab_iter.readline()
        while line is not '':
            if line not in vocab_list:
                if i is not 0:
                    cnt_add+=1
                vocab_list.append(line)
            line = vocab_iter.readline()

print(cnt_add, 9644-9357)
vocab = './data/yelp/vocab'
vocab_write = open(vocab, "w")

for word in vocab_list:
    vocab_write.writelines(word)