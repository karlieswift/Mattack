"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: adv.py
@Desc:
"""
import os

import numpy as np
import torch
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import warnings
from tqdm import tqdm
import json
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

from PIL import Image

paths = 'vit_gpt2_coco_model2'
path1 = './images'
for path in os.listdir(paths):
    print(path)#model
    answers=[]
    pt_save_directory = os.path.join(paths, path)
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
    for pic_path in tqdm(os.listdir(path1)):
        
        pic_name=""
        pic_path1 = pic_path.split('.')[0]
        index = 0
        for i in range(len(pic_path1)):
            if pic_path1[i] == "0":
                continue
            pic_name=pic_path1[i:]
            break
            
            
            
        pokemon = Image.open(os.path.join(path1, pic_path))
        pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        answers.append({pic_name:generated_text})
    with open('./eval_answer/answer2_{}.json'.format(path), 'w') as fp:
        json.dump(answers, fp)



