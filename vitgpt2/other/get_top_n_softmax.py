"""
@Env: /anaconda3/python3.10
@Time: 2023/8/2-13:50
@Auth: karlieswift
@File: get_top_n_softmax.py
@Desc: 
"""
import os
import pandas as pd
import numpy as np
import torch
from ...attackutils import myGlobal
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
import warnings
from tqdm import tqdm
import json

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

myGlobal._init()
from PIL import Image
number=1
paths = 'vit_gpt2_coco_model{}'.format(number)
path1 = './natural_images6'
class_path= './eval_data_coco1000'
# from transforevamers.generation.attackutils impo
for path in os.listdir(paths):
    if 'CE' not in path:
        continue
    print(path)  # model
    answers = []
    pt_save_directory = os.path.join(paths, path)
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
    for image_path in tqdm(os.listdir(class_path)):
        myGlobal.set_value('soft_max_index', [])
        image = Image.open(os.path.join(class_path,image_path))
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        soft_max=myGlobal.get_value('soft_max_index')

        answers.append([path,image_path.split('.')[0],generated_text,soft_max])
    df=pd.DataFrame(answers,columns=["model_loss","image_name","generated_text",'soft_max'])

    df.to_csv('./eval_answer_softmax/answer_sotmax{}_{}.csv'.format(number,path),index=False)




