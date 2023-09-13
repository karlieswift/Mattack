"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: eval_caption.py
@Desc:
"""


import os

import torch
from blip2opt.models import load_model_and_preprocess
from PIL import Image
from attackutils import myGlobal
myGlobal._init()

#paths='./eval_pics'
paths='/public/home/mswanghao/image_caption/natural_images100/airplane'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

finetune_paths=[
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_CE/20230819153/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_6/20230819142/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_10/20230819144/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_20/20230819144/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output3/BLIP2/Caption_coco_drsl_0_100/20230819145/checkpoint_0.pth'
]
for finetune_path in finetune_paths:
    myGlobal.set_value("finetune_path", finetune_path)
    if 'CE' not in finetune_path:
        b = 1e-5
        loss_name = "DRSL3"
        start = int(finetune_path.split('/')[-3].split('_')[-2])
        end = int(finetune_path.split('/')[-3].split('_')[-1])
        myGlobal.set_value('b', b)
        myGlobal.set_value('loss_name', loss_name)
        myGlobal.set_value('start', start)
        myGlobal.set_value('end', end)
        print("loss {} b={} start={} end={}".format(loss_name, b, start, end))
    else:
        myGlobal.set_value('loss_name', "CE")
        print("loss {}".format("CE"))
   # model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True,device=device)
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True,device=device)
    model=model.half()
    for path in os.listdir(paths)[:4]:
        image_path = os.path.join(paths, path)
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        answers = model.generate({"image": image})
        print("图片：{} 回答：{}".format(path, answers))
    del model
    del vis_processors

