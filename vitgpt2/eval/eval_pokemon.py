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

ds = datasets.load_dataset(
    '/public/home/mswanghao/.cache/huggingface/datasets/arrow/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec-fe76e6d8a6bb4c5d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137')
print(ds)
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
index=100
print(train_ds[index])
pokemon=train_ds['image'][index]
#text=train_ds['text'][0]

myGlobal._init()
from PIL import Image
pokemon=Image.open('../images/image1.jpg')
#pokemon=Image.open('pokemon2.png')
print("DRSL1")
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel
device = "cuda" if torch.cuda.is_available() else "cpu"
pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_pokemon_model/model_DRSL1"
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
pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_pokemon_model/model_CE"
print("加载预训练")
model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("预测:",generated_text1)


