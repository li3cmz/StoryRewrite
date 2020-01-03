import numpy as np
from bert_score import score

folder = 'out_1010_EvolveGTAE_base6' ###modify
epoch = 10 ###modify
x_gt_pred_file = '../../../{}/samples/val_x.{}'.format(folder, epoch)
yy1_gt_pred_file = '../../../{}/samples/val_y1.{}'.format(folder, epoch)
yy2_gt_pred_file = '../../../{}/samples/val_y2.{}'.format(folder, epoch)
yy3_gt_pred_file = '../../../{}/samples/val_y3.{}'.format(folder, epoch)


def load_f(x_gt_pred_file, yy1_gt_pred_file, yy2_gt_pred_file, yy3_gt_pred_file):
    refs_yy = []
    hyps_yy = []
    with open(x_gt_pred_file,'r') as x_f, open(yy1_gt_pred_file,'r') as yy1_f,  open(yy2_gt_pred_file,'r') as yy2_f, open(yy3_gt_pred_file,'r') as yy3_f:
        x_gt = x_f.readline().strip()
        x_pred = x_f.readline().strip()
        yy1_gt = yy1_f.readline().strip()
        yy1_pred = yy1_f.readline().strip()
        yy2_gt = yy2_f.readline().strip()
        yy2_pred = yy2_f.readline().strip()
        yy3_gt = yy3_f.readline().strip()
        yy3_pred = yy3_f.readline().strip()
        
        while yy1_gt is not '':
            refs_yy.append(x_gt)
            hyps_yy.append(x_pred)
            refs_yy.append(yy1_gt)
            hyps_yy.append(yy1_pred)
            refs_yy.append(yy2_gt)
            hyps_yy.append(yy2_pred)
            refs_yy.append(yy3_gt)
            hyps_yy.append(yy3_pred)
            
            x_gt = x_f.readline().strip()
            x_pred = x_f.readline().strip()
            yy1_gt = yy1_f.readline().strip()
            yy1_pred = yy1_f.readline().strip()
            yy2_gt = yy2_f.readline().strip()
            yy2_pred = yy2_f.readline().strip()
            yy3_gt = yy3_f.readline().strip()
            yy3_pred = yy3_f.readline().strip()

    return hyps_yy, refs_yy


def bert_score(x_gt_pred_file, yy1_gt_pred_file, yy2_gt_pred_file, yy3_gt_pred_file):

    cands, refs = load_f(x_gt_pred_file, yy1_gt_pred_file, yy2_gt_pred_file, yy3_gt_pred_file)
    P, R, F = score(cands, refs, bert="bert-base-uncased")#"pytorch-transformers")
    return P, R, F

if __name__ == "__main__":
    P, R, F = bert_score(x_gt_pred_file, yy1_gt_pred_file, yy2_gt_pred_file, yy3_gt_pred_file)
    
    print("mean_P: {}".format( np.nanmean(np.array(P))))
    print("mean_R: {}".format( np.nanmean(np.array(R))))
    print("mean_F: {}".format( np.nanmean(np.array(F))))
