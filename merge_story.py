import os
import sys
import json
import fileinput
### for bert pretrained
bert_pretrained = False  ###modify for lower or not
out_chaochu = True ###modify

oper=int(sys.argv[1])
folder = sys.argv[2]
epoch=int(sys.argv[3])

if oper==1:
    y1_file = './{}/samples/val_y1.{}'.format(folder, epoch) #yy_gt and yy_pred
    y2_file = './{}/samples/val_y2.{}'.format(folder, epoch) #yy_gt and yy_pred
    y3_file = './{}/samples/val_y3.{}'.format(folder, epoch) #yy_gt and yy_pred
    x_file = './{}/samples/val_x.{}'.format(folder, epoch)
    yy1_gt_y1 = './{}/samples/val_yy1gt_y1.{}'.format(folder, epoch) #yy_gt and y_gt
    yy2_gt_y2 = './{}/samples/val_yy2gt_y2.{}'.format(folder, epoch) #yy_gt and y_gt
    yy3_gt_y3 = './{}/samples/val_yy3gt_y3.{}'.format(folder, epoch) #yy_gt and y_gt

    #把标点符号合并进句子#[',','?','!','.']，去除bert的##分词符号
    pro_file_list = [y1_file, y2_file, y3_file, x_file, yy1_gt_y1, yy2_gt_y2, yy3_gt_y3]
    for file_ in pro_file_list:
        for line in fileinput.input(file_,inplace = True):
            line = line.strip().replace(" .",".")
            line = line.strip().replace(" ,",",")
            line = line.strip().replace(" ?","?")
            line = line.strip().replace(" !","!")
            line = line.strip().replace(" ##","")
            while line.find("..") != -1:
                line = line.strip().replace("..",".")
            print(line)


    # save all endings as dict 使用x1做为key, no重复
    dev_data = './data/TimeTravel-no-filter/dev_data.json'
    test_data = './data/TimeTravel-no-filter/test_data.json'
    lookup_dict = {}
    import json
    with open(dev_data,'r') as load_f:
        line = load_f.readline()
        while line is not '':
            line_dict = json.loads(line)
            if bert_pretrained:
                x1 = line_dict['premise'][:-1].lower()
                for i in range(len(line_dict['edited_endings'])):
                    for j in range(len(line_dict['edited_endings'][i])):
                        line_dict['edited_endings'][i][j] = line_dict['edited_endings'][i][j].lower()
            else:
                x1 = line_dict['premise'][:-1]
            
            lookup_dict[x1] = line_dict['edited_endings']
            line = load_f.readline()

    with open(test_data,'r') as load_f:
        line = load_f.readline()
        while line is not '':
            line_dict = json.loads(line)
            if bert_pretrained:
                x1 = line_dict['premise'][:-1].lower()
                for i in range(len(line_dict['edited_endings'])):
                    for j in range(len(line_dict['edited_endings'][i])):
                        line_dict['edited_endings'][i][j] = line_dict['edited_endings'][i][j].lower()
            else:
                x1 = line_dict['premise'][:-1]
            lookup_dict[x1] = line_dict['edited_endings']
            line = load_f.readline()



    # 创造评估文件
    unfound=0
    story_file_pred_dir = './Evaluation_model/StoryTask/results/{}'.format(folder)
    story_file_gt_dir = './Evaluation_model/StoryTask/results/{}'.format(folder)
    if not os.path.exists(story_file_pred_dir):
        os.makedirs(story_file_pred_dir)
    if not os.path.exists(story_file_gt_dir):
        os.makedirs(story_file_gt_dir)
    story_file_pred = open('{}/story{}_pred.txt'.format(story_file_pred_dir, epoch),'w')
    story_file_gt = open('{}/story{}_gt.jsonl'.format(story_file_gt_dir, epoch),'w')

    import re
    write_num=0
    with open(y1_file,'r') as y1_f, open(y2_file,'r') as y2_f, open(y3_file,'r') as y3_f, open(x_file,'r') as x_f, open(yy1_gt_y1, 'r') as yy1_gt_y1_f, open(yy2_gt_y2, 'r') as yy2_gt_y2_f, open(yy3_gt_y3, 'r') as yy3_gt_y3_f:
        yy1_gt = y1_f.readline().strip()
        yy1_pred = y1_f.readline().strip()
        yy2_gt = y2_f.readline().strip()
        yy2_pred = y2_f.readline().strip()
        yy3_gt = y3_f.readline().strip()
        yy3_pred = y3_f.readline().strip()
        x1x2 = x_f.readline().strip()
        x1xx2 = x_f.readline().strip()

        if out_chaochu:
            yy1_pred = re.split("\. |\? |! ", yy1_pred)[0]+'.'
            yy2_pred = re.split("\. |\? |! ", yy2_pred)[0]+'.'
            yy3_pred = re.split("\. |\? |! ", yy3_pred)[0]+'.'

        x1 = re.split("\.|!|\?", x1x2)[0].strip()

        yy1_gt_y1_f.readline().strip() ##去掉yy1_gt,前面已经读了
        y1_gt = yy1_gt_y1_f.readline().strip() #y1_gt
        yy2_gt_y2_f.readline().strip()
        y2_gt = yy2_gt_y2_f.readline().strip() #y2_gt
        yy3_gt_y3_f.readline().strip()
        y3_gt = yy3_gt_y3_f.readline().strip() #y3_gt

        while yy1_gt is not '':
            story_o = {}
            story_o['ori_context'] = x1x2
            story_o['cf_context'] = x1xx2
            story_o['ori_endinng'] = y1_gt + ' ' + y2_gt + ' ' + y3_gt
            
            if x1 not in lookup_dict:
                unfound+=1
                #print('unfound: ', unfound)
                yy1_gt = y1_f.readline().strip()
                yy1_pred = y1_f.readline().strip()
                yy2_gt = y2_f.readline().strip()
                yy2_pred = y2_f.readline().strip()
                yy3_gt = y3_f.readline().strip()
                yy3_pred = y3_f.readline().strip()
                x1x2 = x_f.readline().strip()
                x1xx2 = x_f.readline().strip()

                if out_chaochu:
                    #if "the bus was late, and if she didn ' t start walking soon, she ' d be late." in yy1_pred:
                    yy1_pred = re.split("\. |\? |! ", yy1_pred)[0]+'.'
                    yy2_pred = re.split("\. |\? |! ", yy2_pred)[0]+'.'
                    yy3_pred = re.split("\. |\? |! ", yy3_pred)[0]+'.'

                x1 = re.split("\.|!|\?", x1x2)[0].strip()
                
                yy1_gt_y1_f.readline().strip() ##去掉yy1_gt,前面已经读了
                y1_gt = yy1_gt_y1_f.readline().strip() #y1_gt
                yy2_gt_y2_f.readline().strip()
                y2_gt = yy2_gt_y2_f.readline().strip() #y2_gt
                yy3_gt_y3_f.readline().strip()
                y3_gt = yy3_gt_y3_f.readline().strip() #y3_gt

                continue

            story_o['gold_end'] = lookup_dict[x1]
            story_o_str = json.dumps(story_o)
            
            story_c = yy1_pred + ' ' + yy2_pred + ' ' + yy3_pred

            story_file_gt.writelines([story_o_str+'\n'])
            story_file_pred.writelines([story_c+'\n'])


            yy1_gt = y1_f.readline().strip()
            yy1_pred = y1_f.readline().strip()
            yy2_gt = y2_f.readline().strip()
            yy2_pred = y2_f.readline().strip()
            yy3_gt = y3_f.readline().strip()
            yy3_pred = y3_f.readline().strip()
            x1x2 = x_f.readline().strip()
            x1xx2 = x_f.readline().strip()

            if out_chaochu:
                yy1_pred = re.split("\. |\? |! ", yy1_pred)[0]+'.'
                yy2_pred = re.split("\. |\? |! ", yy2_pred)[0]+'.'
                yy3_pred = re.split("\. |\? |! ", yy3_pred)[0]+'.'

            x1 = re.split("\.|!|\?", x1x2)[0].strip()
            
            yy1_gt_y1_f.readline().strip() ##去掉yy1_gt,前面已经读了
            y1_gt = yy1_gt_y1_f.readline().strip() #y1_gt
            yy2_gt_y2_f.readline().strip()
            y2_gt = yy2_gt_y2_f.readline().strip() #y2_gt
            yy3_gt_y3_f.readline().strip()
            y3_gt = yy3_gt_y3_f.readline().strip() #y3_gt


else:
    story_file_pred_dir = './Evaluation_model/StoryTask/results/{}/story{}_pred.txt'.format(folder, epoch)

    for line in fileinput.input(story_file_pred_dir,inplace = True):
        line = line.strip().replace(" .",".")
        line = line.strip().replace(" ,",",")
        line = line.strip().replace(" ?","?")
        line = line.strip().replace(" !","!")
        line = line.strip().replace(" ##","")
        
        while line.find("..") != -1:
            line = line.strip().replace("..",".")
        while line.find("!!") != -1:
            line = line.strip().replace("!!","!")
        while line.find("!.") != -1:
            line = line.strip().replace("!.","!")
        while line.find(".!") != -1:
            line = line.strip().replace(".!","!")
        print(line)