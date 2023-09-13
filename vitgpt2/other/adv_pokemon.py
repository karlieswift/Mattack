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
warnings.filterwarnings("ignore")


import os

def del_files2(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))





del_files2('../adv_pics')


myGlobal._init()
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")


myGlobal.set_value('b', 0.00001)
myGlobal.set_value('top_n', 2)
loss_name='DRSL1'
print(loss_name)
if loss_name=='CE':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_pokemon_model/model_CE"
elif loss_name=='DRSL':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_pokemon_model/model_DRSL"
elif loss_name=='DRSL1':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_pokemon_model/model_DRSL1"

model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)



model.to(device)
MAX_PATCHES = 1024
alpha = 0.01
epsilon = 0.6


image = Image.open('../images/pokemon1.jpg')
orgin_image = None

for epoch in range(1, 200):
    if epoch > 1:
        image = Image.open('./adv_pics/{}.jpg'.format(epoch - 1))
    labels = tokenizer(
        "Young people in the rain.",
        return_tensors="pt",
    ).input_ids.to(device)
    print("labels:",labels)
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    loss = -model(pixel_values=pixel_values, labels=labels).loss

    loss.backward()
    # print("loss",loss)
    images = myGlobal.get_value("image")
    if epoch == 1:
        orgin_image = images
    grad = images.grad
    adv_images = images + alpha * grad.sign()
    # adv_images=torch.clamp(adv_images,min=a.min(),max=a.max())
    adv_images = torch.max(torch.min(adv_images, orgin_image + epsilon), orgin_image - epsilon)

    from attackutils.save_pic import tensor2pic1,tensor2pic2

    tensor2pic2(adv_images.detach().cpu(), './adv_pics/{}.jpg'.format(epoch))
    model.eval()
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("第{}次攻击: 生成内容:{}".format(epoch,generated_text))
