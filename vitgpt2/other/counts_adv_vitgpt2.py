"""
@Env: /anaconda3/python3.10
@Time: 2023/7/30-17:13
@Auth: karlieswift
@File: 攻击代码.py
@Desc:
"""

import torch
from ...attackutils import myGlobal
import pandas as pd
import os
import warnings
import time
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize

warnings.filterwarnings("ignore")
start = time.time()


def del_files2(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


# stop_words = ['man', 'woman', 'people', 'umbrellas', 'couple', 'group']#目标攻击
# stop_words = ['jet','plane', 'jetliner', 'flying', 'air', 'sky', 'flying','airplane']#非目标攻击

myGlobal._init()
from PIL import Image
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

device = "cuda" if torch.cuda.is_available() else "cpu"

aa = str(time.time())[:10]
files_adv = './' + aa
if not os.path.exists(files_adv):
    os.makedirs(files_adv)
    
    
    
number=2
model_path = 'vit_gpt2_coco_model{}'.format(number)
dict_class_pic_path = {'dog': ['dog', 'cat', 'animal'], 'cat': ['dog', 'cat', 'animal'],
                       'car': ['car', 'parked', 'motorcycle', 'truck'],
                       'airplane': ['jet', 'passenger', 'airport', 'jetliner', 'plane', 'propeller', 'airplane']}
for class_path, stop_words in dict_class_pic_path.items():
    loss_names = ['CE', 'DRSL', 'DRSL1_2', 'DRSL1_6', 'DRSL1_10']
    res_all = []
    for loss_name in loss_names:
        print(loss_name)
        if loss_name == 'CE':
            myGlobal.set_value('loss_name', "CE")
            myGlobal.set_value('top_n', 2)
            pt_save_directory = "./{}/model_CE".format(model_path)
        elif loss_name == 'DRSL':
            myGlobal.set_value('loss_name', "DRSL")
            myGlobal.set_value('top_n', 2)
            myGlobal.set_value('b', 1e-8)
            pt_save_directory = "./{}/model_DRSL".format(model_path)
        elif loss_name == 'DRSL1_2':
            myGlobal.set_value('loss_name', "DRSL1")
            myGlobal.set_value('top_n', 2)
            myGlobal.set_value('b', 0.00001)
            pt_save_directory = "./{}/model_DRSL1_2".format(model_path)
        elif loss_name == 'DRSL1_6':
            myGlobal.set_value('loss_name', "DRSL1")
            myGlobal.set_value('top_n', 6)
            myGlobal.set_value('b', 0.00001)
            pt_save_directory = "./{}/model_DRSL1_6".format(model_path)
        elif loss_name == 'DRSL1_10':
            myGlobal.set_value('loss_name', "DRSL1")
            myGlobal.set_value('top_n', 10)
            myGlobal.set_value('b', 0.00001)
            pt_save_directory = "./{}/model_DRSL1_10".format(model_path)

        model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
        tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
        image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)

        model.to(device)

        # alpha = 0.02
        # epsilon = 1
        alpha = 0.01
        epsilon = 0.6
        orgin_image = None

        index_counts = 1
        for p1 in os.listdir(os.path.join("./natural_images6", class_path)):
            orgin_generated_text = ""
            score = 1
            del_files2(files_adv)
            print(loss_name, index_counts)
            if index_counts > 100:
                break
            index_counts += 1
            image_path = os.path.join('./natural_images6', class_path, p1)
            image = Image.open(image_path)
            adv_res = []  # 记录每次攻击的图片回答的内容
            meteor_socres = []  # 记录每次攻击的图片回答的内容
            generated_ids_list=[]
            softmax_top_n=[]
            for epoch in range(1, 22):
                myGlobal.set_value('soft_max_index', [])
                if epoch > 1:
                    image = Image.open('./{}/{}.jpg'.format(files_adv, epoch - 1))
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
                from attackutils.save_pic import tensor2pic1, tensor2pic2

                tensor2pic2(adv_images.detach().cpu(), '{}/{}.jpg'.format(files_adv, epoch))
                adv_res.append(generated_text)

                meteor_socres.append(score)
                generated_ids_list.append(generated_ids.cpu().tolist())
                soft_max_index = myGlobal.get_value('soft_max_index')
                softmax_top_n.append(soft_max_index)
                flag = False
                i = 0
                while i < len(stop_words):
                    if stop_words[i] in generated_text:
                        break
                    i += 1
                    if i == len(stop_words):
                        flag = False

                if flag:
                    break

            res_all.append([loss_name, p1, adv_res, generated_ids_list,softmax_top_n])
    pd.DataFrame(res_all, columns=['loss_name', 'pic_name', 'answers',"generated_ids_list","softmax_top_n"]).to_csv(
        './adv_vitgpt2/adv_{}_{}.csv'.format(number,class_path),
        index=False)

print(time.time() - start)

