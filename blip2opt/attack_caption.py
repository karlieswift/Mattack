"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: attack_caption.py
@Desc:
"""

import torch
import time
import os
from blip2opt.models import load_model_and_preprocess
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from attackutils import myGlobal
from attackutils.save_pic import tensor2pic2
myGlobal._init()

def del_files(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))


def tensor2pic(x, path, std, mean, dpi=300):
    """
    x:tensor.shape=(3,224,224)  x 为标准化的图片
    path: save path pics
    dpi:像素密度
    """
    x = x.mul(torch.FloatTensor(std).view(3, 1, 1)).add(torch.FloatTensor(mean).view(3, 1, 1)).numpy()
    im = np.transpose(x, (1, 2, 0))
    fig, ax = plt.subplots()
    ax.imshow(im, aspect='equal')
    plt.axis('off')
    height, width, channels = im.shape
    fig.set_size_inches(width / dpi, height / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(path, dpi=dpi)


mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)


finetune_paths=[
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_CE/20230819153/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_6/20230819142/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_10/20230819144/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_20/20230819144/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_100/20230819145/checkpoint_0.pth'
]
paths = '/public/home/mswanghao/image_caption/natural_images1000'
#paths = '/public/home/mswanghao/.cache/blip2opt/coco/images/train2014'
random.seed(666)
all_images = random.sample(os.listdir(paths), 200)
for finetune_path in finetune_paths[4:]:
    myGlobal.set_value("finetune_path", finetune_path)
    if 'CE' not in finetune_path:
        b = 1e-5
        loss_name = "DRSL3"
        start = int(finetune_path.split('/')[-3].split('_')[-2])
        end = int(finetune_path.split('/')[-3].split('_')[-1])
        file_adv_csv_name="natural_caption_{}_{}_{}".format(loss_name,str(start),str(end))
        myGlobal.set_value('b', b)
        myGlobal.set_value('loss_name', loss_name)
        myGlobal.set_value('start', start)
        myGlobal.set_value('end', end)
        print("loss {} b={} start={} end={}".format(loss_name, b, start, end))
    else:
        loss_name = "CE"
        file_adv_csv_name = "natural_caption_{}".format(loss_name)
        myGlobal.set_value('loss_name', loss_name)
        print("loss {}".format("CE"))

    aa = str(time.time())[:10]
    files_adv = './' + aa
    if not os.path.exists(files_adv):
        os.makedirs(files_adv)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    #model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
    alpha = 0.5
    epsilon = 1
    res_all=[]
    start_time=time.time()
    index=1
    model=model.half()
    for path in tqdm(all_images):
        print(path)
        if index>1000:
            break
        index+=1
        image_path = os.path.join(paths, path)
        del_files(files_adv)
        n_dim=len(np.array(Image.open(image_path)).shape)
        if n_dim!=3:
            continue
        image = Image.open(image_path)
        image = vis_processors["eval"](image).unsqueeze(0).to(device)
        orgin_image = image
        adv_images = None
        adv_answers=[]
        for epoch in range(1, 12):
            if epoch > 1:
                image = adv_images
            image.requires_grad = True
            model.eval()
            # answers = model.generate({"image": image, "prompt": "Question: Is this a airplane? Answer:"})
            answers = model.generate({"image": image})
            print("第{}次攻击: 生成内容:{}".format(epoch, answers[0]))
            # samples = {'image': image, 'text_input': ["A plane shot across the sky."]}
            samples = {'image': image, 'text_input': answers}
            loss = model(samples)["loss"]
            loss.backward()
            grad = image.grad
            adv_images = image + alpha * grad.sign()
            adv_images = torch.max(torch.min(adv_images, orgin_image + epsilon), orgin_image - epsilon)
            # tensor2pic(x=adv_images.detach().cpu().squeeze(0), path='{}/{}.jpg'.format(files_adv, epoch), std=std,           mean=mean)
            adv_images = adv_images.detach()
            adv_answers.append(answers[0])
            del image
            torch.cuda.empty_cache()

        res_all.append([loss_name,path,adv_answers])
    pd.DataFrame(res_all,columns=['loss_name',"image",'answers']).to_csv('./adv_files/{}.csv'.format(file_adv_csv_name),index=False)
    del model
    del vis_processors
print(time.time()-start_time)
