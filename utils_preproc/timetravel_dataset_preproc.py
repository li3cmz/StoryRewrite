import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(3) ### modify

# split dataset to little file
'''
train_supervised_large_y_filter_line_number=[223, 242, 316, 325, 444, 460, 474, 626, 674, 1054, 1152, 1232, 1275, 1352, 1414, 1454, 1483, 1560, 1747, 1792, 1800, 1829, 1838, 2289, 2316, 2383, 2450, 2463, 2524, 2571, 2909, 2961, 3075, 3189, 3333, 3500, 3803, 4051, 4084, 4112, 4134, 4396, 4436, 4497, 5195, 5386, 5407, 5456, 5691, 5957, 6657, 6736, 6741, 6771, 6817, 6836, 6980, 7369, 7460, 7463, 7481, 7500, 8135, 8169, 8200, 9011, 9166, 9568, 9710, 9847, 9931, 10429, 10455, 10758, 11016, 11086, 11135, 11223, 11308, 11325, 11385, 11702, 12079, 12978, 13007, 13165, 13222, 13248, 13254, 13283, 13327, 13391, 13403, 13440, 13527, 13535, 13578, 13907, 13938, 13959, 13985, 14118, 14203, 14221, 14503, 14612, 14677, 14760, 14797, 14866, 14877, 15001, 15050, 15089, 15102, 15254, 15355, 15588, 15696, 15753, 15772, 15898, 15924, 15962, 16008, 16072, 16094, 16109, 16143, 16209, 16234, 16369, 16370, 16502, 16610, 16706, 16777, 16958, 17308, 17362, 17366, 17453, 17601, 17788, 17843, 18014, 18190, 18252, 18343, 18450, 18454, 18507, 18606, 18693, 18725, 18729, 18762, 18788, 18802, 19035, 19103, 19133, 19186, 19222, 19262, 19431, 19614, 19879, 20777, 21402, 21910, 22043, 22453, 23123, 23957, 24027, 24281, 24449, 25119, 25651, 25957, 26033, 26101, 26388, 26847, 26904, 26964, 27136, 27243, 27261, 27501, 27637, 27737, 27846, 28089, 28190]
train_supervised_small_y_filter_line_number=[223, 242, 316, 325, 444, 460, 474, 626, 674, 1054, 1152, 1232, 1275, 1352, 1414, 1454, 1483, 1560, 1747, 1792, 1800, 1829, 1838, 2289, 2316, 2383, 2450, 2463, 2524, 2571, 2909, 2961, 3075, 3189, 3333, 3500, 3803, 4051, 4084, 4112, 4134, 4216, 4221, 4251, 4297, 4316, 4460, 4849, 4940, 4943, 4961, 4980, 5145, 5403, 5473, 5522, 5610, 5695, 5712, 5772, 6089, 6466, 7365, 7394, 7552, 7609, 7635, 7641, 7670, 7714, 7778, 7790, 7827, 7914, 7922, 7965, 8294, 8325, 8346, 8372, 8505, 8590, 8608, 8890, 8999, 9064, 9147, 9184, 9253, 9264, 9388, 9437, 9476, 9489, 9641, 9742, 9975, 10084, 10141, 10160, 10286, 10312, 10350, 10396, 10460, 10482, 10497, 10531, 10597, 10622, 10757, 10758, 10890, 10998, 11094, 11165, 11346, 11696, 11750, 11754, 11841, 11990, 12177, 12232, 12403, 12579, 12641, 12732, 12839, 12843, 12896, 12995, 13082, 13114, 13118, 13151, 13177, 13191, 13424, 13492, 13522, 13575, 13611, 13651, 13820, 14003, 14268, 15166, 15791, 16299, 16432]
train_unsupervised_y_filter_line_number=[]
test_data_y_filter_line_number=[67, 138, 154, 366, 388, 623, 799, 1121, 1184, 1209, 1279, 1297, 1375, 1467, 1549, 1554, 1716]
dev_data_y_filter_line_number= [103, 136, 290, 315, 358, 416, 614, 740, 894, 1088, 1129, 1445, 1463, 1690, 1828]
filter_line_dict = {}
filter_line_dict['train_supervised_large'] = train_supervised_large_y_filter_line_number
filter_line_dict['train_supervised_small'] = train_supervised_small_y_filter_line_number
filter_line_dict['train_unsupervised'] = train_unsupervised_y_filter_line_number
filter_line_dict['test_data'] = test_data_y_filter_line_number
filter_line_dict['dev_data'] = dev_data_y_filter_line_number
train_x1 = open('../data/TimeTravel/pre/dev_data_x1.txt', 'w')
train_x2 = open('../data/TimeTravel/pre/dev_data_x2.txt', 'w')
train_xx2 = open('../data/TimeTravel/pre/dev_data_xx2.txt', 'w')
train_y = open('../data/TimeTravel/pre/dev_data_y.txt', 'w')
train_yy = open('../data/TimeTravel/pre/dev_data_yy.txt', 'w') # only end1
train_yy1_end1 = open('../data/TimeTravel/pre/dev_data_yy1_end1.txt', 'w')
train_yy2_end1 = open('../data/TimeTravel/pre/dev_data_yy2_end1.txt', 'w')
train_yy3_end1 = open('../data/TimeTravel/pre/dev_data_yy3_end1.txt', 'w')
train_yy1_end2 = open('../data/TimeTravel/pre/dev_data_yy1_end2.txt', 'w')
train_yy2_end2 = open('../data/TimeTravel/pre/dev_data_yy2_end2.txt', 'w')
train_yy3_end2 = open('../data/TimeTravel/pre/dev_data_yy3_end2.txt', 'w')
train_yy1_end3 = open('../data/TimeTravel/pre/dev_data_yy1_end3.txt', 'w')
train_yy2_end3 = open('../data/TimeTravel/pre/dev_data_yy2_end3.txt', 'w')
train_yy3_end3 = open('../data/TimeTravel/pre/dev_data_yy3_end3.txt', 'w')
train_yy_endings = open('../data/TimeTravel/pre/dev_data_yy_endings.txt', 'w')
#train_y = open('../data/TimeTravel/pre/dev_data_y_for_split.txt', 'w')
cur_line_number = 0
train_file = '../data/TimeTravel-no-filter/dev_data.json'
with open(train_file,'r') as load_f:
    line = load_f.readline()
    while line is not '':
        cur_line_number+=1
        #print(cur_line_number)
        if cur_line_number in filter_line_dict['dev_data']:
            print("filter number: ", cur_line_number)
            line = load_f.readline()
            continue
        line_dict = json.loads(line)
        x1 = line_dict['premise']
        x2 = line_dict['initial']
        xx2 = line_dict['counterfactual']
        y = line_dict['original_ending']
        yy = line_dict['edited_endings'] #[[],[],[]]
        
        train_x1.writelines([x1+'\n'])
        train_x2.writelines([x2+'\n'])
        train_xx2.writelines([xx2+'\n'])
        train_y.writelines([y+'\n'])
        
        #train_yy.writelines([yy_str+'\n'])
        train_yy1_end1.writelines([yy[0][0]+'\n'])
        train_yy2_end1.writelines([yy[0][1]+'\n'])
        train_yy3_end1.writelines([yy[0][2]+'\n'])
        train_yy1_end2.writelines([yy[1][0]+'\n'])
        train_yy2_end2.writelines([yy[1][1]+'\n'])
        train_yy3_end2.writelines([yy[1][2]+'\n'])
        train_yy1_end3.writelines([yy[2][0]+'\n'])
        train_yy2_end3.writelines([yy[2][1]+'\n'])
        train_yy3_end3.writelines([yy[2][2]+'\n'])
        
        train_yy_endings.write(str(yy))
        train_yy_endings.write('\n')
        print(cur_line_number)
        line = load_f.readline()
'''

# split y.txt to y1,y2andy3
'''
import re
train_y1 = open('../data/TimeTravel/pre/train_supervised_large_y1.txt', 'w')
train_y2 = open('../data/TimeTravel/pre/train_supervised_large_y2.txt', 'w')
train_y3 = open('../data/TimeTravel/pre/train_supervised_large_y3.txt', 'w')
train_file = '../data/TimeTravel(no filter)/train_supervised_large_y_for_split.txt'
cnt = 1
len_less_than_3_lineNumber_list = []
line_num = 0
with open(train_file,'r') as load_f:
    line = load_f.readline()
    while line is not '':
        line_list = re.split("\|\|\|", line)
        if len(line_list)<3 :
            len_less_than_3_lineNumber_list.append(cnt)
            print(line)
            cnt+=1
            line = load_f.readline()
            continue
        
        y1 = line_list[0].strip()
        y2 = line_list[1].strip()
        y3 = line_list[2].strip()
        train_y1.writelines([y1+'\n'])
        train_y2.writelines([y2+'\n'])
        train_y3.writelines([y3+'\n'])
        line_num+=1
        #print("line_num: ", line_num)
        line = load_f.readline()
        cnt+=1
print("len_less_than_3_lineNumber_list_len: ",len(len_less_than_3_lineNumber_list))
print("file: ", train_file, "line_number: ", len_less_than_3_lineNumber_list)
'''

# 合并数据
'''#生成x1x2 和x1xx2
dataset = ['train_supervised_large','train_supervised_small', 'train_unsupervised', 'dev_data','test_data']
for mode in dataset:
    x1_file = '../data/TimeTravel/pre/{}_x1.txt'.format(mode)
    x2_file = '../data/TimeTravel/pre/{}_x2.txt'.format(mode)
    x1x2_file = '../data/TimeTravel/pre/{}_x1x2.text'.format(mode)
    x1x2_f = open(x1x2_file, 'w')
    with open(x1_file,'r') as x1_f, open(x2_file,'r') as x2_f:
        x1 = x1_f.readline()
        x2 = x2_f.readline()
        while x1 is not '':
            x1x2 = x1.strip() + ' ' + x2.strip()
            x1x2_f.writelines([x1x2+'\n'])
            x1 = x1_f.readline()
            x2 = x2_f.readline()
'''


# 生成 x1x2 ||| x1xx2
# 和   y1 , y2 , y3 , yy1 , yy2, yy3
'''
dataset = ['train_supervised_large','train_supervised_small',  'dev_data','test_data']
for mode in dataset:
    x1x2_file = '../data/TimeTravel/pre/{}_x1x2.text'.format(mode)
    x1xx2_file = '../data/TimeTravel/pre/{}_x1xx2.text'.format(mode)
    y1_file = '../data/TimeTravel/pre/{}_y1.txt'.format(mode)
    y2_file = '../data/TimeTravel/pre/{}_y2.txt'.format(mode)
    y3_file = '../data/TimeTravel/pre/{}_y3.txt'.format(mode)
    yy1_file = '../data/TimeTravel/pre/{}_yy1.txt'.format(mode)
    yy2_file = '../data/TimeTravel/pre/{}_yy2.txt'.format(mode)
    yy3_file = '../data/TimeTravel/pre/{}_yy3.txt'.format(mode)
    
    all_file = '../data/TimeTravel/TimeTravel.{}_x.text'.format(mode)
    all_f = open(all_file, 'w')
    all_file1 = '../data/TimeTravel/TimeTravel.{}_y1_yy1.text'.format(mode)
    all_f1 = open(all_file1, 'w')
    all_file2 = '../data/TimeTravel/TimeTravel.{}_y2_yy2.text'.format(mode)
    all_f2 = open(all_file2, 'w')
    all_file3 = '../data/TimeTravel/TimeTravel.{}_y3_yy3.text'.format(mode)
    all_f3 = open(all_file3, 'w')
    with open(x1x2_file,'r') as x1x2_f, open(x1xx2_file,'r') as x1xx2_f, open(y1_file,'r') as y1_f, open(y2_file,'r') as y2_f, open(y3_file,'r') as y3_f, open(yy1_file,'r') as yy1_f, open(yy2_file,'r') as yy2_f, open(yy3_file,'r') as yy3_f:
        x1x2 = x1x2_f.readline().strip()
        x1xx2 = x1xx2_f.readline().strip()
        y1 = y1_f.readline().strip()
        y2 = y2_f.readline().strip()
        y3 = y3_f.readline().strip()
        yy1 = yy1_f.readline().strip()
        yy2 = yy2_f.readline().strip()
        yy3 = yy3_f.readline().strip()
        while x1x2 is not '':
            all_ = x1x2 + '|||' +x1xx2
            all_f.writelines([all_+'\n'])
            all_1 = y1 + '|||' + yy1
            all_f1.writelines([all_1+'\n'])
            all_2 =  y2 + '|||' + yy2
            all_f2.writelines([all_2+'\n'])
            all_3 = y3 + '|||' + yy3
            all_f3.writelines([all_3+'\n'])
            x1x2 = x1x2_f.readline().strip()
            x1xx2 = x1xx2_f.readline().strip()
            y1 = y1_f.readline().strip()
            y2 = y2_f.readline().strip()
            y3 = y3_f.readline().strip()
            yy1 = yy1_f.readline().strip()
            yy2 = yy2_f.readline().strip()
            yy3 = yy3_f.readline().strip()
'''
# 生成 x1x2 ||| x1xx2 ||| y1 ||| y2 ||| y3 for unsupervised train dataset 
'''
dataset = ['train_unsupervised']
for mode in dataset:
    x1x2_file = '../data/TimeTravel/pre/{}_x1x2.text'.format(mode)
    x1xx2_file = '../data/TimeTravel/pre/{}_x1xx2.text'.format(mode)
    y1_file = '../data/TimeTravel/pre/{}_y1.txt'.format(mode)
    y2_file = '../data/TimeTravel/pre/{}_y2.txt'.format(mode)
    y3_file = '../data/TimeTravel/pre/{}_y3.txt'.format(mode)
    all_file = '../data/TimeTravel/TimeTravel.{}_x.text'.format(mode)
    all_f = open(all_file, 'w')
    all_file2 = '../data/TimeTravel/TimeTravel.{}_y.text'.format(mode)
    all_f2 = open(all_file2, 'w')
    with open(x1x2_file,'r') as x1x2_f, open(x1xx2_file,'r') as x1xx2_f, open(y1_file,'r') as y1_f, open(y2_file,'r') as y2_f, open(y3_file,'r') as y3_f:
        x1x2 = x1x2_f.readline().strip()
        x1xx2 = x1xx2_f.readline().strip()
        y1 = y1_f.readline().strip()
        y2 = y2_f.readline().strip()
        y3 = y3_f.readline().strip()
        while x1x2 is not '':
            all_ = x1x2 + '|||' +x1xx2
            all_2 = y1 + '|||' + y2 + '|||' + y3
            all_f.writelines([all_+'\n'])
            all_f2.writelines([all_2+'\n'])
            x1x2 = x1x2_f.readline().strip()
            x1xx2 = x1xx2_f.readline().strip()
            y1 = y1_f.readline().strip()
            y2 = y2_f.readline().strip()
            y3 = y3_f.readline().strip()
'''

# 统计句子的长度，所有数据集中的最大句长+平均长句
'''
dataset = ['train_supervised_large', 'dev_data','test_data'] #'train_supervised_small'
y_maxlen = -1
xx_maxlen = -1
y_cnt_sentence = 0
y_sentence_lenSum = 0
xx_cnt_sentence = 0
xx_sentence_lenSum = 0
x_cnt_bigger_30 = 0
for mode in dataset:
    x1x2_file = '../data/TimeTravel/pre/{}_x1x2.text'.format(mode)
    x1xx2_file = '../data/TimeTravel/pre/{}_x1xx2.text'.format(mode)
    y1_file = '../data/TimeTravel/pre/{}_y1.txt'.format(mode)
    y2_file = '../data/TimeTravel/pre/{}_y2.txt'.format(mode)
    y3_file = '../data/TimeTravel/pre/{}_y3.txt'.format(mode)
    yy1_file = '../data/TimeTravel/pre/{}_yy1.txt'.format(mode)
    yy2_file = '../data/TimeTravel/pre/{}_yy2.txt'.format(mode)
    yy3_file = '../data/TimeTravel/pre/{}_yy3.txt'.format(mode)
    with open(x1x2_file,'r') as x1x2_f, open(x1xx2_file,'r') as x1xx2_f, open(y1_file,'r') as y1_f, open(y2_file,'r') as y2_f, open(y3_file,'r') as y3_f, open(yy1_file,'r') as yy1_f, open(yy2_file,'r') as yy2_f, open(yy3_file,'r') as yy3_f:
        x1x2 = x1x2_f.readline().strip()
        x1xx2 = x1xx2_f.readline().strip()
        y1 = y1_f.readline().strip()
        y2 = y2_f.readline().strip()
        y3 = y3_f.readline().strip()
        yy1 = yy1_f.readline().strip()
        yy2 = yy2_f.readline().strip()
        yy3 = yy3_f.readline().strip()
        while x1x2 is not '':
            x1x2_len = len(x1x2.split())
            x1xx2_len = len(x1xx2.split())
            y1_len = len(y1.split())
            y2_len = len(y2.split())
            y3_len = len(y3.split())
            yy1_len = len(yy1.split())
            yy2_len = len(yy2.split())
            yy3_len = len(yy3.split())
            len_list = [y1_len, y2_len, y3_len, yy1_len, yy2_len, yy3_len] #x1x2_len, x1xx2_len, 
            max_len = max(len_list)
            if max_len > y_maxlen:
                y_maxlen = max_len
            y_cnt_sentence+=1
            y_sentence_lenSum+=max_len
            len_list = [x1x2_len, x1xx2_len]
            max_len = max(len_list)
            if max_len > 35:
                x_cnt_bigger_30+=1
            if max_len > xx_maxlen:
                xx_maxlen = max_len
            xx_cnt_sentence+=1
            xx_sentence_lenSum+=max_len
            x1x2 = x1x2_f.readline().strip()
            x1xx2 = x1xx2_f.readline().strip()
            y1 = y1_f.readline().strip()
            y2 = y2_f.readline().strip()
            y3 = y3_f.readline().strip()
            yy1 = yy1_f.readline().strip()
            yy2 = yy2_f.readline().strip()
            yy3 = yy3_f.readline().strip()
print("y_maxlen: ", y_maxlen, "avg_sentence_len: ", y_sentence_lenSum/float(y_cnt_sentence))
print("xx_maxlen: ", xx_maxlen, "avg_sentence_len: ", xx_sentence_lenSum/float(xx_cnt_sentence))
print("x_cnt_bigger_30: ", x_cnt_bigger_30)
'''
#y_maxlen:  33 avg_sentence_len:  13.493584716253098
#xx_maxlen:  46 avg_sentence_len:  20.85434639395175 x_cnt_bigger_35: 85

# 接下来对标点符号和单词进行分隔
#For train
'''
dataset = ['train_supervised_large']#, 'train_supervised_large','dev_data','test_data'] #'train_supervised_small' #'train_supervised_large', 'dev_data','test_data' #'train_unsupervised'
global_maxlen = -1
cnt_sentence = 0
sentence_lenSum = 0
fuhao1 = ['  ,','  ?','  !','  .']
fuhao2 = [' ,',' ?',' !',' .']
#fuhao1 = [',']
#fuhao2 = [' ,']
for mode in dataset:
    x1_file = '../data/TimeTravel/pre/{}_x1.txt'.format(mode)
    x2_file = '../data/TimeTravel/pre/{}_x2.txt'.format(mode)
    xx2_file = '../data/TimeTravel/pre/{}_xx2.txt'.format(mode)
    #x1x2_file = '../data/TimeTravel/pre/{}_x1x2.text'.format(mode)
    #x1xx2_file = '../data/TimeTravel/pre/{}_x1xx2.text'.format(mode)
    y_file = '../data/TimeTravel/pre/{}_y.txt'.format(mode)
    #y1_file = '../data/TimeTravel/pre/{}_y1.txt'.format(mode)
    #y2_file = '../data/TimeTravel/pre/{}_y2.txt'.format(mode)
    #y3_file = '../data/TimeTravel/pre/{}_y3.txt'.format(mode)
    yy_file = '../data/TimeTravel/pre/{}_yy.txt'.format(mode)
    yy1_file = '../data/TimeTravel/pre/{}_yy1.txt'.format(mode)
    yy2_file = '../data/TimeTravel/pre/{}_yy2.txt'.format(mode)
    yy3_file = '../data/TimeTravel/pre/{}_yy3.txt'.format(mode)
    x1_ff = open('../data/TimeTravel/pre/{}_x1.txt_1'.format(mode), 'w')
    x2_ff = open('../data/TimeTravel/pre/{}_x2.txt_1'.format(mode), 'w')
    xx2_ff = open('../data/TimeTravel/pre/{}_xx2.txt_1'.format(mode), 'w')
    #x1x2_ff = open('../data/TimeTravel/pre/{}_x1x2.text'.format(mode), 'w')
    #x1xx2_ff = open('../data/TimeTravel/pre/{}_x1xx2.text'.format(mode), 'w')
    y_ff = open('../data/TimeTravel/pre/{}_y.txt_1'.format(mode), 'w')
    #y1_ff = open('../data/TimeTravel/pre/{}_y1.txt_1'.format(mode), 'w')
    #y2_ff = open('../data/TimeTravel/pre/{}_y2.txt_1'.format(mode), 'w')
    #y3_ff = open('../data/TimeTravel/pre/{}_y3.txt_1'.format(mode), 'w')
    yy_ff = open('../data/TimeTravel/pre/{}_yy.txt_1'.format(mode), 'w')
    yy1_ff = open('../data/TimeTravel/pre/{}_yy1.txt_1'.format(mode), 'w')
    yy2_ff = open('../data/TimeTravel/pre/{}_yy2.txt_1'.format(mode), 'w')
    yy3_ff = open('../data/TimeTravel/pre/{}_yy3.txt_1'.format(mode), 'w')
    
    with open(x1_file,'r') as x1_f, open(x2_file,'r') as x2_f, open(xx2_file,'r') as xx2_f, open(y_file,'r') as y_f, open(yy1_file,'r') as yy1_f, open(yy2_file,'r') as yy2_f, open(yy3_file,'r') as yy3_f, open(yy_file,'r') as yy_f:
        x1 = x1_f.readline().strip()
        x2 = x2_f.readline().strip()
        xx2 = xx2_f.readline().strip()
        #x1x2 = x1x2_f.readline().strip()
        #x1xx2 = x1xx2_f.readline().strip()
        y = y_f.readline().strip()
        #y1 = y1_f.readline().strip()
        #y2 = y2_f.readline().strip()
        #y3 = y3_f.readline().strip()
        yy = yy_f.readline().strip()
        yy1 = yy1_f.readline().strip()
        yy2 = yy2_f.readline().strip()
        yy3 = yy3_f.readline().strip()
        while x1 is not '':
            for idx in range(len(fuhao1)):
                x1 = x1.replace(fuhao1[idx],fuhao2[idx])
                x2 = x2.replace(fuhao1[idx],fuhao2[idx])
                xx2 = xx2.replace(fuhao1[idx],fuhao2[idx])
                #x1x2 = x1x2.replace(fuhao1[idx],fuhao2[idx])
                #x1xx2 = x1xx2.replace(fuhao1[idx],fuhao2[idx])
                y = y.replace(fuhao1[idx],fuhao2[idx])
                #y1 = y1.replace(fuhao1[idx],fuhao2[idx])
                #y2 = y2.replace(fuhao1[idx],fuhao2[idx])
                #y3 = y3.replace(fuhao1[idx],fuhao2[idx])
                yy = yy.replace(fuhao1[idx],fuhao2[idx])
                yy1 = yy1.replace(fuhao1[idx],fuhao2[idx])
                yy2 = yy2.replace(fuhao1[idx],fuhao2[idx])
                yy3 = yy3.replace(fuhao1[idx],fuhao2[idx])
            
            print(x1)
            x1_ff.writelines([x1+'\n'])
            x2_ff.writelines([x2+'\n'])
            xx2_ff.writelines([xx2+'\n'])
            #x1x2_ff.writelines([x1x2+'\n'])
            #x1xx2_ff.writelines([x1xx2+'\n'])
            y_ff.writelines([y+'\n'])
            #y1_ff.writelines([y1+'\n'])
            #y2_ff.writelines([y2+'\n'])
            #y3_ff.writelines([y3+'\n'])
    
            yy_ff.writelines([yy+'\n'])
            yy1_ff.writelines([yy1+'\n'])
            yy2_ff.writelines([yy2+'\n'])
            yy3_ff.writelines([yy3+'\n'])
          
            x1 = x1_f.readline().strip()
            x2 = x2_f.readline().strip()
            xx2 = xx2_f.readline().strip()
            #x1x2 = x1x2_f.readline().strip()
            #x1xx2 = x1xx2_f.readline().strip()
            y = y_f.readline().strip()
            #y1 = y1_f.readline().strip()
            #y2 = y2_f.readline().strip()
            #y3 = y3_f.readline().strip()
           
            yy = yy_f.readline().strip()
            yy1 = yy1_f.readline().strip()
            yy2 = yy2_f.readline().strip()
            yy3 = yy3_f.readline().strip()
'''
#将文件名中的_1去掉


# For dev and test
'''
dataset = ['dev_data','test_data']#, 'train_supervised_large', 'dev_data','test_data'] #'train_supervised_small' #'train_supervised_large', 'dev_data','test_data' #'train_unsupervised'
global_maxlen = -1
cnt_sentence = 0
sentence_lenSum = 0
fuhao1 = [',','?','!','.']
fuhao2 = [' ,',' ?',' !',' .']
#fuhao1 = [',']
#fuhao2 = [' ,']
for mode in dataset:
    x1_file = '../data/TimeTravel/pre/{}_x1.txt'.format(mode)
    x2_file = '../data/TimeTravel/pre/{}_x2.txt'.format(mode)
    xx2_file = '../data/TimeTravel/pre/{}_xx2.txt'.format(mode)
    y_file = '../data/TimeTravel/pre/{}_y.txt'.format(mode)
    yy1_file_end1 = '../data/TimeTravel/pre/{}_yy1_end1.txt'.format(mode)
    yy2_file_end1 = '../data/TimeTravel/pre/{}_yy2_end1.txt'.format(mode)
    yy3_file_end1 = '../data/TimeTravel/pre/{}_yy3_end1.txt'.format(mode)
    yy1_file_end2 = '../data/TimeTravel/pre/{}_yy1_end2.txt'.format(mode)
    yy2_file_end2 = '../data/TimeTravel/pre/{}_yy2_end2.txt'.format(mode)
    yy3_file_end2 = '../data/TimeTravel/pre/{}_yy3_end2.txt'.format(mode)
    yy1_file_end3 = '../data/TimeTravel/pre/{}_yy1_end3.txt'.format(mode)
    yy2_file_end3 = '../data/TimeTravel/pre/{}_yy2_end3.txt'.format(mode)
    yy3_file_end3 = '../data/TimeTravel/pre/{}_yy3_end3.txt'.format(mode)
    x1_ff = open('../data/TimeTravel/pre/{}_x1.txt_1'.format(mode), 'w')
    x2_ff = open('../data/TimeTravel/pre/{}_x2.txt_1'.format(mode), 'w')
    xx2_ff = open('../data/TimeTravel/pre/{}_xx2.txt_1'.format(mode), 'w')
    y_ff = open('../data/TimeTravel/pre/{}_y.txt_1'.format(mode), 'w')
    yy1_ff_end1 = open('../data/TimeTravel/pre/{}_yy1_end1.txt_1'.format(mode), 'w')
    yy2_ff_end1 = open('../data/TimeTravel/pre/{}_yy2_end1.txt_1'.format(mode), 'w')
    yy3_ff_end1 = open('../data/TimeTravel/pre/{}_yy3_end1.txt_1'.format(mode), 'w')
    yy1_ff_end2 = open('../data/TimeTravel/pre/{}_yy1_end2.txt_1'.format(mode), 'w')
    yy2_ff_end2 = open('../data/TimeTravel/pre/{}_yy2_end2.txt_1'.format(mode), 'w')
    yy3_ff_end2 = open('../data/TimeTravel/pre/{}_yy3_end2.txt_1'.format(mode), 'w')
    yy1_ff_end3 = open('../data/TimeTravel/pre/{}_yy1_end3.txt_1'.format(mode), 'w')
    yy2_ff_end3 = open('../data/TimeTravel/pre/{}_yy2_end3.txt_1'.format(mode), 'w')
    yy3_ff_end3 = open('../data/TimeTravel/pre/{}_yy3_end3.txt_1'.format(mode), 'w')
    
    with open(x1_file,'r') as x1_f, open(x2_file,'r') as x2_f, open(xx2_file,'r') as xx2_f, open(y_file,'r') as y_f, open(yy1_file_end1,'r') as yy1_f_end1, \
        open(yy2_file_end1,'r') as yy2_f_end1, open(yy3_file_end1,'r') as yy3_f_end1, open(yy1_file_end2,'r') as yy1_f_end2, \
        open(yy2_file_end2,'r') as yy2_f_end2, open(yy3_file_end2,'r') as yy3_f_end2, open(yy1_file_end3,'r') as yy1_f_end3, \
        open(yy2_file_end3,'r') as yy2_f_end3, open(yy3_file_end3,'r') as yy3_f_end3:
        x1 = x1_f.readline().strip()
        x2 = x2_f.readline().strip()
        xx2 = xx2_f.readline().strip()
       
        y = y_f.readline().strip()
        
        yy1_end1 = yy1_f_end1.readline().strip()
        yy2_end1 = yy2_f_end1.readline().strip()
        yy3_end1 = yy3_f_end1.readline().strip()
        yy1_end2 = yy1_f_end2.readline().strip()
        yy2_end2 = yy2_f_end2.readline().strip()
        yy3_end2 = yy3_f_end2.readline().strip()
        yy1_end3 = yy1_f_end3.readline().strip()
        yy2_end3 = yy2_f_end3.readline().strip()
        yy3_end3 = yy3_f_end3.readline().strip()
        while x1 is not '':
            for idx in range(len(fuhao1)):
                x1 = x1.replace(fuhao1[idx],fuhao2[idx])
                x2 = x2.replace(fuhao1[idx],fuhao2[idx])
                xx2 = xx2.replace(fuhao1[idx],fuhao2[idx])
                y = y.replace(fuhao1[idx],fuhao2[idx])
                yy1_end1 = yy1_end1.replace(fuhao1[idx],fuhao2[idx])
                yy2_end1 = yy2_end1.replace(fuhao1[idx],fuhao2[idx])
                yy3_end1 = yy3_end1.replace(fuhao1[idx],fuhao2[idx])
                yy1_end2 = yy1_end2.replace(fuhao1[idx],fuhao2[idx])
                yy2_end2 = yy2_end2.replace(fuhao1[idx],fuhao2[idx])
                yy3_end2 = yy3_end2.replace(fuhao1[idx],fuhao2[idx])
                yy1_end3 = yy1_end3.replace(fuhao1[idx],fuhao2[idx])
                yy2_end3 = yy2_end3.replace(fuhao1[idx],fuhao2[idx])
                yy3_end3 = yy3_end3.replace(fuhao1[idx],fuhao2[idx])
            
            print(x1)
            x1_ff.writelines([x1+'\n'])
            x2_ff.writelines([x2+'\n'])
            xx2_ff.writelines([xx2+'\n']) 
            y_ff.writelines([y+'\n'])
            
    
            yy1_ff_end1.writelines([yy1_end1+'\n'])
            yy2_ff_end1.writelines([yy2_end1+'\n'])
            yy3_ff_end1.writelines([yy3_end1+'\n'])
            yy1_ff_end2.writelines([yy1_end2+'\n'])
            yy2_ff_end2.writelines([yy2_end2+'\n'])
            yy3_ff_end2.writelines([yy3_end2+'\n'])
            yy1_ff_end3.writelines([yy1_end3+'\n'])
            yy2_ff_end3.writelines([yy2_end3+'\n'])
            yy3_ff_end3.writelines([yy3_end3+'\n'])
          
            x1 = x1_f.readline().strip()
            x2 = x2_f.readline().strip()
            xx2 = xx2_f.readline().strip()
            y = y_f.readline().strip()
            yy1_end1 = yy1_f_end1.readline().strip()
            yy2_end1 = yy2_f_end1.readline().strip()
            yy3_end1 = yy3_f_end1.readline().strip()
            yy1_end2 = yy1_f_end2.readline().strip()
            yy2_end2 = yy2_f_end2.readline().strip()
            yy3_end2 = yy3_f_end2.readline().strip()
            yy1_end3 = yy1_f_end3.readline().strip()
            yy2_end3 = yy2_f_end3.readline().strip()
            yy3_end3 = yy3_f_end3.readline().strip()
'''
#将文件名中的_1去掉


mode_list=['dev_data','test_data', 'train_supervised_large'] ###modify train_supervised_large or train_data
folder='../data/TimeTravel/pre'
prefix='' #
folder_res = '../data/TimeTravel/baby_model5'
for mode in mode_list:
    x1_file='{}/{}{}_x1.txt'.format(folder, prefix, mode)
    x1x2_file='{}/{}{}_x1x2.text'.format(folder, prefix, mode)
    x1xx2_file='{}/{}{}_x1xx2.text'.format(folder, prefix, mode)
    y_file='{}/{}{}_y.txt'.format(folder, prefix, mode)
    yy1_file='{}/{}{}_yy1.txt'.format(folder, prefix, mode)
    yy2_file='{}/{}{}_yy2.txt'.format(folder, prefix, mode)
    yy3_file='{}/{}{}_yy3.txt'.format(folder, prefix,  mode)
    mask_text = 'Unknown .'

    res_f = open('{}/{}_ctx.text'.format(folder_res, mode), 'w')
    with open(x1_file, 'r') as x1_f, open(x1x2_file, 'r') as x1x2_f, open(x1xx2_file, 'r') as x1xx2_f, open(y_file, 'r') as y_f, \
        open(yy1_file, 'r') as yy1_f, open(yy2_file, 'r') as yy2_f, open(yy3_file, 'r') as yy3_f: 
        x1 = x1_f.readline().strip()
        x1x2 = x1x2_f.readline().strip()
        x1xx2 = x1xx2_f.readline().strip()
        y = y_f.readline().strip()
        yy1 = yy1_f.readline().strip()
        yy2 = yy2_f.readline().strip()
        yy3 = yy3_f.readline().strip()


        while x1x2 is not '':
            x1x2yx1xx2 = x1x2 + ' ' + y + ' | ' + x1xx2
            x1x2yx1xx2yy = x1x2 + ' ' + y + ' | ' + x1xx2 + ' ' + yy1 + ' ' + yy2 + ' ' + yy3
            x1x2y = x1x2 + ' ' + y
            x1 = x1
            x1x2yx1my = x1x2 + ' ' + y + ' | ' + x1 + ' ' + mask_text + ' ' + y
            x1x2yx1m = x1x2 + ' ' + y + ' | ' + x1 + ' ' + mask_text
            ctx = x1x2 + '|||' + x1xx2 + '|||' + x1x2yx1xx2 + '|||' + x1x2yx1xx2yy + '|||' + \
                x1x2y + "|||" + x1 + "|||" + x1x2yx1my + "|||" + x1x2yx1m

            res_f.writelines([ctx+'\n'])

            x1 = x1_f.readline().strip()
            x1x2 = x1x2_f.readline().strip()
            x1xx2 = x1xx2_f.readline().strip()
            y = y_f.readline().strip()
            yy1 = yy1_f.readline().strip()
            yy2 = yy2_f.readline().strip()
            yy3 = yy3_f.readline().strip()