import os

import random
from tqdm import tqdm
import pandas as pd
import torch
from blip2opt.models import load_model_and_preprocess
from PIL import Image
from attackutils import myGlobal
myGlobal._init()




paths = '/public/home/mswanghao/image_caption/natural_images1000'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
import os

# pretrain_path="https://storage.googleapis.com/sfr-vision-language-research/blip2opt/models/BLIP2/blip2_pretrained_opt2.7b.pth"
pretrain_path='/public/home/mswanghao/TorchProject/blip2opt/blip2opt/output_vqa/BLIP2/CE/20230820094/checkpoint_1.pth'
myGlobal.set_value("pretrain_path", pretrain_path)

loss_name = "CE"
myGlobal.set_value('loss_name', loss_name)
print("loss {}".format("CE"))

model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True,device=device)
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True,device=device)

random.seed(1)
samples=random.sample(os.listdir(paths),1000)
res_all=[]
for ip in tqdm(samples):
    image_path = os.path.join(paths, ip)
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    #answers = model.generate({"image": image, "prompt": "Is this a picture about {} ?. Answer:".format(targets)})
    # answers = model.generate({"image": image})#caption
    answers = model.generate({"image": image, "prompt": "Write a short description for the image. Answer:"})
    res_all.append(['blip2_pretrained_opt2.7b',ip,answers[0]])
    # print(res_all[-1])

pd.DataFrame(res_all,columns=['model',"image",'answer']).to_csv('../eval_natural1000.csv', index=False)
del model
del vis_processors




