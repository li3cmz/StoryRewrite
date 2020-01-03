## For all samples:
### Step one:
content_preservation and naturalness can eval directly.
### Step two:
python get_sample_label_origin_and_trans.py得到分离的origin.text, origin.labels和trans.txt, trans.labels
### Step three:
python BertScore/calculate.py
### Step four:
python ACCUEMD.py



## For 1000 samples:
### Step one:
python to_sample_1000_result.py 得到1000个样本的origin和trans
### Step two:
content_preservation and naturalness can eval directly.
### Step three:
python get_sample_label_origin_and_trans.py得到分离的origin.text, origin.labels和trans.txt, trans.labels
### Step four:
python ACCUEMD.py