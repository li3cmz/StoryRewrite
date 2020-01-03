import sys
import json

# for our own model's result 测量我们的模型的结果WMS和s+wms指标的数据处理
folder = sys.argv[1] ###modify
epoch=int(sys.argv[2]) ###modify
sample_1800=False ###modify 选取1800个样本出来评估
pred = './results/{}/story{}_pred.txt'.format(folder,epoch)
gt = './results/{}/story{}_gt.jsonl'.format(folder,epoch)

if sample_1800: ##modify
    output_file = open('./results/{}/smd_pair_{}_1800.tsv'.format(folder,epoch),'w') ##modify
    max_cnt = 1800 ##modify
else:
    output_file = open('./results/{}/smd_pair_{}.tsv'.format(folder,epoch),'w')
    max_cnt = 100000 
cnt=0
with open(pred, 'r') as pred_f, open(gt, 'r') as gt_f:
    line = pred_f.readline().strip()
    line_gt = gt_f.readline().strip()
    cnt+=1
    while line is not '' and cnt < max_cnt:
        line_gt_dict = json.loads(line_gt)
        gt_endings = line_gt_dict['gold_end']
        
        pred_end_str = line
        for gt_end_list in gt_endings:
            gt_end_str = " ".join(gt_end_list).strip()
            output_file.writelines([gt_end_str+'\t'+pred_end_str+'\n'])
            break ###modify only care one gt endings
        line = pred_f.readline().strip()
        line_gt = gt_f.readline().strip()
        cnt+=1


# for Hui Liang's lunwen result, preprocess for all metrics
## get yy_gt_pred.txt
'''
import fileinput
import re
name = 'test_samples_x1x2yx1xx2_10000' ###modify
file_ = './results/gpt-2/{}.tsv'.format(name)
file_yy_gt = './results/gpt-2/test_samples_x1x2yx1xx2_10000_yy_gt.tsv' ###modify


for line in fileinput.input(file_,inplace = True):
        line = line.strip().replace("\t"," ")
        line = line.strip().replace(" | ","\t")
        print(line)

for line in fileinput.input(file_yy_gt,inplace = True):
        line = line.strip().replace(" | ","\t")
        print(line)


# get lookup 3endings dict
dev_data = '../../data/TimeTravel-no-filter/dev_data.json'
test_data = '../../data/TimeTravel-no-filter/test_data.json'
lookup_dict = {}
import json
with open(dev_data,'r') as load_f:
    line = load_f.readline()
    while line is not '':
        line_dict = json.loads(line)
        yy = line_dict['edited_endings']
        x1 = line_dict['premise'] #x1
        #print(yy[0][0])
        lookup_dict[x1[:-1]] = yy
        line = load_f.readline()
with open(test_data,'r') as load_f:
    line = load_f.readline()
    while line is not '':
        line_dict = json.loads(line)
        yy = line_dict['edited_endings']
        x1 = line_dict['premise'] #x1
        #print(yy[0][0])
        lookup_dict[x1[:-1]] = yy
        line = load_f.readline()


res = open('./results/gpt-2/{}_yy_gt_pred.tsv'.format(name),'w')
story_file_pred = open('./results/gpt-2/{}_yy_pred.txt'.format(name),'w')
story_file_gt = open('./results/gpt-2/{}_yy_gt.jsonl'.format(name),'w')
short = 0
with open(file_, 'r') as file_f, open(file_yy_gt, 'r') as yy_gt_f:
    line = file_f.readline().strip()
    line_yy_gt = yy_gt_f.readline().strip()
    cnt=1
    while line is not '':
        sentence_2 = line.split("\t")
        yy_gt_sentence = line_yy_gt.split("\t")[1] #x1xx2yy_gt
        x1xx2yy_gt = re.split('\. |! |\? ', yy_gt_sentence)#[2:]
        yy_gt = '. '.join(x1xx2yy_gt[2:]).strip()
    
        s1 = re.split('\. |! |\? ', sentence_2[0]) #x1x2y
        s2 = re.split('\. |! |\? ', sentence_2[1]) #x1xx2yy_pred
        yy_pred = '. '.join(s2[2:5]).strip()
        #print(s1[0],s1[1],'----', s2[0],s2[1], '----', x1xx2yy_gt[0], x1xx2yy_gt[1])

        res.writelines([yy_gt+'\t'+yy_pred+'\n'])
        story_file_pred.writelines([yy_pred + '. '+'\n']) ## get story_file_pred   story_c = yy1 + ' ' + yy2 + ' ' + yy3



        story_o = {}
        story_o['ori_context'] = s1[0] + '. ' + s1[1] + '.'
        story_o['cf_context'] = s2[0] + '. ' + s2[1] + '.'
        story_o['ori_endinng'] = '. '.join(s1[2:]).strip()

        x1 = s1[0]
        if x1 not in lookup_dict:
            short+=1
            print(short)
            line = file_f.readline().strip()
            line_yy_gt = yy_gt_f.readline().strip()
            cnt+=1  
            continue
        
        story_o['gold_end'] = lookup_dict[x1]
        story_o_str = json.dumps(story_o)
        story_file_gt.writelines([story_o_str+'\n'])
        



        line = file_f.readline().strip()
        line_yy_gt = yy_gt_f.readline().strip()
        cnt+=1
'''