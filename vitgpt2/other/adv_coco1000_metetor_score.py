"""
@Env: /anaconda3/python3.10
@Time: 2023/7/26-14:42
@Auth: karlieswift
@File: 攻击.py
@Desc:
"""

from PIL import Image
import torch
from attackutils import myGlobal
from attackutils.save_pic import tensor2pic2
import os
import warnings
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import pandas as pd
import time
from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel
from attackutils.image_processing_vit import ViTImageProcessor
myGlobal._init()


start=time.time()
warnings.filterwarnings("ignore")


def del_files2(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))




device = "cuda" if torch.cuda.is_available() else "cpu"

myGlobal.set_value('b', 0.0001)  # DRSL2
myGlobal.set_value('top_n', 6)  # DRSL2

# myGlobal.set_value('b', 0.0000000001)#DRSL1
aa=str(time.time())[:10]
files_adv='./'+aa
if not os.path.exists(files_adv):
    os.makedirs(files_adv)

loss_names = ['CE', 'DRSL1']

for loss_name in loss_names:
    print(loss_name)
    if loss_name == 'CE':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_CE"
    elif loss_name == 'DRSL':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSLv2"
    elif loss_name == 'DRSL1':
        myGlobal.set_value('loss_name', loss_name)
        pt_save_directory = "/public/home/mswanghao/image_caption/vit_gpt2_coco_model/model_DRSL1_3"
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)

    model.to(device)
    MAX_PATCHES = 1024

    alpha = 0.02
    epsilon = 1

    # image = Image.open('human.jpg')
    orgin_image = None

    pic_path = '/public/home/mswanghao/image_caption/coco2017_adv'
    index_counts = 1
    res_all = []

    #for p1 in os.listdir(pic_path)[:3333]:
    #for p1 in os.listdir(pic_path)[3333:6666]:
    for p1 in os.listdir(pic_path)[:1000]:
        orgin_generated_text = ""
        score = 1
        del_files2(files_adv)
        print(loss_name, index_counts)
        if index_counts > 4000:
            break
        index_counts += 1

        image_path = os.path.join(pic_path, p1)
        image = Image.open(image_path)

        adv_res = []  # 记录每次攻击的图片回答的内容
        meteor_socres = []  # 记录每次攻击的图片回答的内容
        for epoch in range(1, 11):
            if epoch > 1:
                image = Image.open('./{}/{}.jpg'.format(files_adv,epoch - 1))

            pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
            model.eval()
            generated_ids = model.generate(pixel_values)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if epoch == 1:
                orgin_generated_text = generated_text
                print("第{}次攻击: 生成内容:{}".format(epoch, orgin_generated_text))
            else:
                reference = [word_tokenize(orgin_generated_text)]
                candidate = word_tokenize(generated_text)
                score = round(meteor_score(reference, candidate), 4)
                print("第{}次攻击: 生成内容:{}, meteor_score:{}".format(epoch, generated_text, score))

            labels = tokenizer(
                orgin_generated_text,
                return_tensors="pt",
            ).input_ids.to(device)

            # loss = -model(pixel_values=pixel_values, labels=labels).loss#目标攻击
            loss = model(pixel_values=pixel_values, labels=labels).loss  # 非目标攻击

            loss.backward()
            images = myGlobal.get_value("image")
            if epoch == 1:
                orgin_image = images

            grad = images.grad
            adv_images = images + alpha * grad.sign()
            # adv_images=torch.clamp(adv_images,min=a.min(),max=a.max())
            adv_images = torch.max(torch.min(adv_images, orgin_image + epsilon), orgin_image - epsilon)



            tensor2pic2(adv_images.detach().cpu(), '{}/{}.jpg'.format(files_adv,epoch))
            adv_res.append(generated_text)
            meteor_socres.append(score)

        res_all.append([p1, adv_res, meteor_socres])
    pd.DataFrame(res_all, columns=['pic_name', 'answers', "meteor_socres"]).to_csv(
        './adv_no_target_coco1000_meteor_score_{}_{}.csv'.format(loss_name,aa),
        index=False)




print(time.time()-start)

