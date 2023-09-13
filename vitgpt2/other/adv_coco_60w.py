"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: adv.py
@Desc: 
"""
import torch
from attackutils import myGlobal
from PIL import Image
from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel

from transformers import ViTImageProcessor
from attackutils.image_processing_vit import ViTImageProcessor
from attackutils.save_pic import tensor2pic2

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


device = "cuda" if torch.cuda.is_available() else "cpu"

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

model.to(device)
MAX_PATCHES = 1024

myGlobal.set_value('loss_name','CE')
alpha = 0.01
epsilon = 0.6


image = Image.open('../images/dog_and_cat.jpg')
orgin_image = None

for epoch in range(1, 1000):
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



    tensor2pic2(adv_images.detach().cpu(), './adv_pics/{}.jpg'.format(epoch))
    model.eval()
    generated_ids = model.generate(pixel_values)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("第{}次攻击: 生成内容:{}".format(epoch,generated_text))
