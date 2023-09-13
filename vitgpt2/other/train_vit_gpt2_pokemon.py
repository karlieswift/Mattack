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

ds = datasets.load_dataset(
    '/public/home/mswanghao/.cache/huggingface/datasets/arrow/2a3b91fbd88a2c90d1dbbb32b460cf621d31bd5b05b934492fdef7d8d6f236ec-fe76e6d8a6bb4c5d/0.0.0/74f69db2c14c2860059d39860b1f400a03d11bf7fb5a8258ca38c501c878c137')
print(ds)
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]
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
        encoding["text"] = tokenizer(item["text"], max_length=60,  padding="max_length",return_tensors="pt", ).input_ids.to(device)

        encoding["image"] = image_processor(item["image"], return_tensors="pt").pixel_values.to(device)
        return encoding


def collator(batch):
    new_batch = {"images": [], "texts": []}
    for item in batch:
        new_batch["images"].append(item["image"])
        new_batch["texts"].append(item["text"])
    new_batch["images"] = torch.concat(new_batch["images"],dim=0)
    new_batch["texts"] = torch.concat(new_batch["texts"],dim=0)
    return new_batch


train_dataset = ImageCaptioningDataset(train_ds)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

print("训练前")
model.eval()
pokemon = train_ds['image'][0]
pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
generated_ids = model.generate(pixel_values)
generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text1)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

orgin_image = None
progress_Bar = ProgressBar()
'''
"./pokemon_model/model1":CE
"./pokemon_model/model2":原来的DRSL-非目标
"./pokemon_model/model3":现在的DRSL-最大值
'''

myGlobal.set_value('b', 0.00001)
myGlobal.set_value('top_n', 2)
name='DRSL1'
print(name)
if name=='CE':
    myGlobal.set_value('loss_name', 'CE')
    pt_save_directory = "./vit_gpt2_pokemon_model/model_CE"

elif name=="DRSL":
    myGlobal.set_value('loss_name', 'DRSL')
    pt_save_directory = "./vit_gpt2_pokemon_model/model_DRSL"

elif name=="DRSL1":
    myGlobal.set_value('loss_name', 'DRSL1')
    pt_save_directory = "./vit_gpt2_pokemon_model/model_DRSL1"
    
    
    
for epoch in range(1,10):
    for idx, batch in enumerate(train_dataloader):
        pixel_values = batch['images']
        labels = batch['texts']
        loss = model(pixel_values=pixel_values, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_Bar.update(max_value=len(train_dataloader), current_value=idx + 1,
                            info="当前Epoch{}的batch的loss:{:.4f}".format(epoch, loss.item()))
        if idx % 100 == 0:
            model.eval()
            pixel_values = image_processor(pokemon, return_tensors="pt").pixel_values.to(device)
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
