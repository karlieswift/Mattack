"""
@Env: /anaconda3/python3.10
@Time: 2023/7/26-14:42
@Auth: karlieswift
@File: 攻击.py
@Desc:
"""

import numpy as np
import torch
from ...attackutils import myGlobal
import os
import warnings
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import pandas as pd
import time

start=time.time()
warnings.filterwarnings("ignore")



myGlobal._init()
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

myGlobal.set_value('b', 0.0001)  # DRSL2
myGlobal.set_value('top_n', 6)  # DRSL2

# myGlobal.set_value('b', 0.0000000001)#DRSL1
aa=str(time.time())[:10]
files_adv='./'+aa
if not os.path.exists(files_adv):
    os.makedirs(files_adv)

loss_names = ['CE', 'DRSL1']
dirfiles='./counts_eval_natural_images'
res_all = []
for loss_name in loss_names:
    print(loss_name)
    if loss_name == 'CE':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_CE"
    elif loss_name == 'DRSL':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSLv2"
    elif loss_name == 'DRSL1':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSL1_3"
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)

    model.to(device)


    pic_paths = './natural_images'
    for p in os.listdir(pic_paths):
        pic_path=os.path.join(pic_paths,p)
        index_counts = 1
        
        for p1 in os.listdir(pic_path):
            orgin_generated_text = ""
            score = 1
            if index_counts > 100:
                break
            index_counts += 1
    
            image_path = os.path.join(pic_path, p1)
            image = Image.open(image_path)
    
            pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
            print(p,p1, generated_text)
     
    
    
            res_all.append([p,p1, generated_text])
pd.DataFrame(res_all, columns=['class','pic_name', 'answers']).to_csv('./counts_eval_natural_images.csv',index=False)




print(time.time()-start)

