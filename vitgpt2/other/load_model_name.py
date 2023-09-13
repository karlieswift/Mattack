import torch

model=torch.load('/public/home/mswanghao/image_caption/vit_gpt2_coco_model3/model_CE_addsoftmax1/pytorch_model.bin')
for name, param in model.items():
    if name=='decoder.lm_head.weight' or name=='encoder.encoder.layer.0.attention.attention.query.weight':
        print(name,param.shape,param)
