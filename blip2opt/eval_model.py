"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: eval_model.py
@Desc:
"""

import os
import random

import pandas as pd
import torch
from blip2opt.models import load_model_and_preprocess
from PIL import Image
# setup device to use
from attackutils import myGlobal
myGlobal._init()
from tqdm import tqdm


targets=["airplane","car","cat","dog","flower","fruit","motorbike","person"]
paths='/public/home/mswanghao/.cache/lavis/coco/images/val2014'
import json
coco_karpathy_val='/public/home/mswanghao/.cache/lavis/coco/annotations/coco_karpathy_val.json'
with open(coco_karpathy_val,'r') as f:
    data=json.load(f)

all_dict={}
for ana in data:
    all_dict[ana["image"].split('/')[-1]]=ana["caption"][0]

random.seed(666)
samples=random.sample(os.listdir(paths),10000)

aaaa=[]
for s in samples:
    if s in list(all_dict.keys()):
        aaaa.append(s)
    if len(aaaa)>1000:
        break
samples=aaaa
print(len(samples))




device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
import os

look_files = 'checkpoint_1.pth'
def get_list(path,look_files):
    all_res=[]
    paths = os.walk(path)
    for path, dir_lst, file_lst in paths:
        if look_files in file_lst:
            all_res.append(os.path.join(path, look_files))
    return all_res

pretrain_paths=get_list('/public/home/mswanghao/TorchProject/lavis/lavis/output_vqaDRSL',look_files)
print(pretrain_paths)

all_answers=[]
for pretrain_path in pretrain_paths:
    myGlobal.set_value("pretrain_path", pretrain_path)
    if 'CE' not in pretrain_path:
        b = 1e-5
        loss_name = "DRSL3"
        start=int(pretrain_path.split('/')[-3].split('_')[-2])
        end=int(pretrain_path.split('/')[-3].split('_')[-1])
        myGlobal.set_value('b', b)
        myGlobal.set_value('loss_name', loss_name)
        myGlobal.set_value('start', start)
        myGlobal.set_value('end', end)
        print("loss {} b={} start={} end={}".format(loss_name, b, start, end))
    else:
        loss_name = "CE"
        myGlobal.set_value('loss_name', loss_name)
        print("loss {}".format("CE"))

    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True,device=device)
    # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True,device=device)
    for image_p in tqdm(samples):
        image_path = os.path.join(paths, image_p)
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        #answers = model.generate({"image": image, "prompt": "Is this a picture about {} ?. Answer:".format(targets)})
        # answers = model.generate({"image": image})#caption
        answers = model.generate({"image": image, "prompt": "Write a short description for the image. Answer:"})
        print(answers)
        all_answers.append([pretrain_path,image_p,answers[0],all_dict.get(image_p,"")])

    del model
    del vis_processors
pd.DataFrame(all_answers,columns=['loss','image','answer1',"answer2"]).to_csv('eval1.csv')



