"""
@Env: /anaconda3/python3.10
@Time: 2023/7/6-20:51
@Auth: karlieswift
@File: eval_vqa.py
@Desc:
"""

import os
import random

import torch
from blip2opt.models import load_model_and_preprocess
from PIL import Image
from attackutils import myGlobal
myGlobal._init()



targets=["airplane","car","cat","dog","flower","fruit","motorbike","person"]
paths='/public/home/mswanghao/image_caption/natural_images'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
path='/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171'
look_files = 'checkpoint'
def get_list(path,look_files):
    all_res=[]
    paths = os.walk(path)
    for path, dir_lst, file_lst in paths:
        if look_files in file_lst:
            all_res.append(os.path.join(path, look_files))
    return all_res
pretrain_paths=get_list(path,look_files)


# pretrain_paths=[
# # '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/outputvqa/BLIP2/CE/20230819205/checkpoint_1.pth',# batchsize=16
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/CE/20230820094/checkpoint_1.pth',
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_6/20230820092/checkpoint_1.pth',
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_10/20230820091/checkpoint_1.pth',
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_20/20230820091/checkpoint_1.pth',
# '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/DRSL3_0_100/20230820090/checkpoint_1.pth',
# ]


pretrain_paths=[
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171/checkpoint_0.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171/checkpoint_1.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171/checkpoint_2.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171/checkpoint_3.pth',
    '/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqacsu/BLIP2/DRSL3_4_0_20/20230823171/checkpoint_4.pth',
 ]
print(pretrain_paths)
all_answers=[]
for pretrain_path in pretrain_paths:
    myGlobal.set_value("pretrain_path", pretrain_path)
    if 'CE' not in pretrain_path:
        b = 1e-4
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


    for targets in os.listdir(paths):

        taeget_path=os.path.join(paths,targets)
        random.seed(666)
        samples=random.sample(os.listdir(taeget_path),3)
        for path in samples:
            image_path = os.path.join(paths,targets, path)
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            #answers = model.generate({"image": image, "prompt": "Is this a picture about {} ?. Answer:".format(targets)})
            # answers = model.generate({"image": image})#caption
            answers = model.generate({"image": image, "prompt": "Write a short description for the image. Answer:"})
            print("图片：{} 回答：{}".format(path, answers))
    del model
    del vis_processors



