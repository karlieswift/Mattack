"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: attack_vqa.py
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
from attackutils.save_pic import tensor2pic2
from attackutils import myGlobal

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



pretrain_paths=[
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/outputvqa/BLIP2/CE/20230819205/checkpoint_1.pth',# batchsize=16
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/CE/20230820094/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_6/20230820092/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_10/20230820091/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_20/20230820091/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_100/20230820090/checkpoint_1.pth',
]






import os
path='/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL'
look_files = 'checkpoint_1.pth'
def get_list(path,look_files):
    all_res=[]
    paths = os.walk(path)
    for path, dir_lst, file_lst in paths:
        if look_files in file_lst:
            all_res.append(os.path.join(path, look_files))
    return all_res
pretrain_paths=get_list(path,look_files)




pretrain_paths=[
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/CE/20230820094/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_6/20230820092/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_10/20230820091/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_20/20230820091/checkpoint_1.pth',
'/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_100/20230820090/checkpoint_1.pth',
]

pretrain_paths = [
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_4_0_6/20230821145/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_4_0_10/20230821150/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_4_0_20/20230821151/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_6_0_20/20230821151/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_6_0_10/20230821152/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqaDRSL/BLIP2/DRSL3_6_0_6/20230821152/checkpoint_1.pth']

print(pretrain_paths)


paths = '/public/home/mswanghao/image_caption/natural_images1000'
#paths = '/public/home/mswanghao/.cache/blip2opt/vg/images/11'

for pretrain_path in pretrain_paths[5:6]:
    myGlobal.set_value("pretrain_path", pretrain_path)
    if 'CE' not in pretrain_path:
        b = 1e-6
        loss_name = "DRSL3"
        start = int(pretrain_path.split('/')[-3].split('_')[-2])
        end = int(pretrain_path.split('/')[-3].split('_')[-1])
        file_adv_csv_name="1e-6vqa_natural_{}_{}_{}".format(loss_name,str(start),str(end))
        myGlobal.set_value('b', b)
        myGlobal.set_value('loss_name', loss_name)
        myGlobal.set_value('start', start)
        myGlobal.set_value('end', end)
        print("loss {} b={} start={} end={}".format(loss_name, b, start, end))
    else:
        loss_name = "CE"
        file_adv_csv_name = "ce_vqa_1natural_{}".format(loss_name)
        myGlobal.set_value('loss_name', loss_name)
        print("loss {}".format("CE"))

    aa = str(time.time())[:10]
    files_adv = './' + aa
    if not os.path.exists(files_adv):
        os.makedirs(files_adv)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
    alpha = 0.3
    epsilon = 1
    res_all=[]
    start_time=time.time()
    index=1
    model=model.half()
    random.seed(1)
    all_images=random.sample(os.listdir(paths),1000)
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
            answers = model.generate({"image": image, "prompt": "Write a short description for the image. Answer:"})
            # answers = model.generate({"image": image}) #caption
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
