1. wms和s+wms只需要考虑一个yy_gt+yy_pred的结果，注意yy_pred要truncated成少于等于3个句子的endings
2. gpt-2的结果复现使用当前目录的wms_data_process.py一键处理，即可开始测所有的指标，用evaluate测BERT\BLEU-4\ROUGE-L，用WMS中的smd.py测WMS和S+WMS
3. BERT-FT暂时不测，没拿到check-points，且觉得不重要
4. 自己的模型跑的数据，用根目录的merge_story.py去处理得到测量BERT\BLEU-4\ROUGE-L能用的数据，然后用当前目录的wms_data_process.py去得到测量WMS和S+WMS需要的数据