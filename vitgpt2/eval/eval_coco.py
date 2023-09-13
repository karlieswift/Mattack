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

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

from PIL import Image

paths = 'vit_gpt2_coco_model2'
for path in os.listdir(paths):
    print(path)
    pt_save_directory = os.path.join(paths, path)
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    model.to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
    pokemon = Image.open('./images/dog_and_cat.jpg')
    pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("预测:", generated_text1)
    #
    # pokemon = Image.open('dog.jpg')
    # pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
    # generated_ids = model.generate(pixel_values)
    # generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("预测:", generated_text1)





