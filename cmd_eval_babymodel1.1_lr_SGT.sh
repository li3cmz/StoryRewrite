OUTPUT=out_1020_baby_model_1.1_addSGT_adjustLr_fromEpoch15
EPOCH=4

python merge_story.py 1 $OUTPUT $EPOCH
python merge_story.py 2 $OUTPUT $EPOCH


cd ./Evaluation_model/StoryTask
python evaluate.py # 此处需要modify evalutae里面需要修改的参数

python wms_data_process.py $OUTPUT $EPOCH
cd WMS/sms/wmd-relax-master/
python smd.py ../../../results/$OUTPUT/smd_pair_$EPOCH.tsv glove wms
python smd.py ../../../results/$OUTPUT/smd_pair_$EPOCH.tsv glove s+wms