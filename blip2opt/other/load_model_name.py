import torch

model=torch.load('/public/home/mswanghao/.cache/torch/hub/checkpoints/blip2_caption_opt2.7b.pth')
for name, param in model['model'].items():
    print(name)
