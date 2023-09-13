"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: adv.py
@Desc:
"""
import numpy as np
import torch
from attackutils  import myGlobal
import warnings
import sys
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

warnings.filterwarnings("ignore")


class ProgressBar:

    def __init__(self, width=30):
        self.width = width

    def update(self, max_value, current_value, info):
        progress = int(round(self.width * current_value / max_value))
        bar = '=' * progress + '.' * (self.width - progress)
        prefix = '{}/{}'.format(current_value, max_value)

        prefix_max_len = len('{}/{}'.format(max_value, max_value))
        buffer = ' ' * (prefix_max_len - len(prefix))

        sys.stdout.write('\r {} {} [{}] - {}'.format(prefix, buffer, bar, info))
        sys.stdout.flush()

    def new_line(self):
        print()


import datasets

from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir='/public/home/mswanghao/TorchProject/vitgpt2/train2017', split="train")
#ds = load_dataset("imagefolder", data_dir='/public/home/mswanghao/TorchProject/pix2Struct/newtrain2017', split="train")
ds = load_dataset('/public/home/mswanghao/.cache/huggingface/datasets/imagefolder/default-0eb6d06ccf55506a/0.0.0/37fbb85cc714a338bea574ac6c7d0b5be5aff46c1862c1989b20e0771199e93f', split="train")
print(ds)

train_ds = ds

print(train_ds[0])

myGlobal._init()
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "/public/home/mswanghao/TorchProject/vitgpt2/gpt2checkpont"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
image_processor = ViTImageProcessor.from_pretrained(image_encoder_model)

tokenizer.pad_token = tokenizer.eos_token
# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

output_dir = "vit-gpt-model-base"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
image_processor.save_pretrained(output_dir)

model.to(device)
MAX_PATCHES = 1024

from torch.utils.data import Dataset, DataLoader


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = {}
        encoding["text"] = tokenizer(item["text"], max_length=128, padding="max_length",
                                     return_tensors="pt", ).input_ids.to(device)

        encoding["image"] = image_processor(item["image"], return_tensors="pt").pixel_values.to(device)
        return encoding


def collator(batch):
    new_batch = {"images": [], "texts": []}
    for item in batch:
        new_batch["images"].append(item["image"])
        new_batch["texts"].append(item["text"])
    new_batch["images"] = torch.concat(new_batch["images"], dim=0)
    new_batch["texts"] = torch.concat(new_batch["texts"], dim=0)
    return new_batch


train_dataset = ImageCaptioningDataset(train_ds)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, collate_fn=collator)

print("训练前")
model.eval()
#pokemon = Image.open('dog_and_cat.jpg')
pokemon=train_ds[0]['image']
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

orgin_image = None
progress_Bar = ProgressBar()
'''
"./vit_gpt2_coco_model/model1":CE
"./vit_gpt2_coco_model/model2":原来的DRSL-非目标
"./vit_gpt2_coco_model/model3":现在的DRSL-最大值
'''
b = 0.00000001
b=0.00001
top_n=2
myGlobal.set_value('b', b)
myGlobal.set_value('top_n', top_n)

name = 'DRSL1'
print(name)
print("b:",b)
print("top_n:",top_n)
if name == 'CE':
    myGlobal.set_value('loss_name', 'CE')
    pt_save_directory = "./vit_gpt2_coco_model1/model_CE"

elif name == "DRSL":
    myGlobal.set_value('loss_name', 'DRSL')
    pt_save_directory = "./vit_gpt2_coco_model1/model_DRSL"

elif name == "DRSL1":
    myGlobal.set_value('loss_name', 'DRSL1')
    pt_save_directory = "./vit_gpt2_coco_model1/model_DRSL1_{}".format(top_n)

for idx, batch in enumerate(train_dataloader):
    pixel_values = batch['images']
    labels = batch['texts']
    loss = model(pixel_values=pixel_values, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    progress_Bar.update(max_value=len(train_dataloader), current_value=idx + 1,
                        info="当前Epoch{}的batch的loss:{:.4f}".format(1, loss.item()))
    if idx % 200 == 0:
        model.eval()
        pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] 
        print(generated_text1)
        dog = Image.open('images/dog_and_cat.jpg')
        pixel_values = image_processor(dog, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text1)
        image_processor.save_pretrained(pt_save_directory)
        model.save_pretrained(pt_save_directory)
        tokenizer.save_pretrained(pt_save_directory)

        model.train()
        progress_Bar.new_line()

print("加载预训练")
model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
model.to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text1)
