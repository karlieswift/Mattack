"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: adv.py
@Desc:
"""
from PIL import Image
import torch
from attackutils import myGlobal
from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel
from attackutils.image_processing_vit import ViTImageProcessor
from attackutils.save_pic import tensor2pic2
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
def del_files2(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))





stop_words=['man','woman','people','umbrellas','couple','group']
myGlobal._init()



device = "cuda" if torch.cuda.is_available() else "cpu"

#myGlobal.set_value('b', 0.0001)
myGlobal.set_value('b', 0.0000000001)
myGlobal.set_value('top_n', 6)

loss_name='DRSL'
print(loss_name)
if loss_name=='CE':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_CE"
elif loss_name=='DRSL':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSLv2"
elif loss_name=='DRSL1':
    myGlobal.set_value('loss_name',loss_name)
    pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSL1_3"
model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)


model.to(device)
MAX_PATCHES = 1024


alpha = 0.01
epsilon = 0.6


# image = Image.open('human.jpg')
orgin_image = None
pic_path='/public/home/mswanghao/image_caption/natural_images/airplane'
index_counts=1
res_all=[]
for p1 in os.listdir(pic_path):
    del_files2('../adv_pics')
    if index_counts>100:
        break
    index_counts+=1

    image_path=os.path.join(pic_path,p1)
    image = Image.open(image_path)

    adv_res=[]#记录每次攻击的图片回答的内容
    for epoch in range(1, 50):
        if epoch > 1:
            image = Image.open('./adv_pics/{}.jpg'.format(epoch - 1))
        labels = tokenizer(
            # "A dog and a cat sitting on a couch.",
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
        adv_res.append(generated_text)
        flag=False
        for s in stop_words:
            if s in generated_text:
                flag=True
                break
        if flag:
            break
                
    res_all.append([p1,adv_res])
pd.DataFrame(res_all,columns=['pic_name','answer']).to_csv('./adv_dog_DRSL1.csv',index=False)


