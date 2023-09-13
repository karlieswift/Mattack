from tqdm import tqdm


"""
@Env: /anaconda3/python3.10
@Time: 2023/7/31-9:11
@Auth: karlieswift
@File: 1.py
@Desc: 
"""
import os
import random
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil
with open('/public/home/mswanghao/TorchProject/vitgpt2/annotations/captions_train2017.json', 'r', encoding='UTF-8') as fp:
    data = json.load(fp)
ids=[image['id'] for image in data['images']]
id2name={image['id']: image['file_name'] for image in data['images']}
path="/public/home/mswanghao/TorchProject/vitgpt2/train2017"




new_annotations=[]
index=1
for id in tqdm(ids):
    temp = []
    image_name=id2name[id]
    image_path=os.path.join(path,image_name)
    n_dim=len(np.array(Image.open(image_path)).shape)
    if n_dim!=3:
        continue
    shutil.copy(image_path,'./coco2017/train2017/')
    for annotations in data['annotations']:
        if annotations['image_id']==id:
            temp.append(annotations['caption'])
    new_annotations.append({id:temp})


with open('./coco2017/captions.json', 'w') as fp:
    json.dump(new_annotations, fp)
