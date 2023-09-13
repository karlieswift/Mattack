"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: adv.py
@Desc:
"""
import numpy as np
import torch
from attackutils import myGlobal
import warnings
import sys
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
warnings.filterwarnings("ignore")


import datasets


myGlobal._init()
from PIL import Image
pokemon=Image.open('../images/dog.jpg')

print("DRSL1")
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
device = "cuda" if torch.cuda.is_available() else "cpu"
pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_flicker_model/model_DRSL1"
print("加载预训练")
model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("预测:",generated_text1)



print("CE")
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
device = "cuda" if torch.cuda.is_available() else "cpu"
pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_flicker_model/model_CE"
print("加载预训练")
model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("预测:",generated_text1)


