"""
@Env: /anaconda3/python3.10
@Time: 2023/5/7-15:51
@Auth: karlieswift
@File: save_pic.py
@Desc: 
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils,transforms


mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]


def tensor2pic1(images,path):
    image=utils.make_grid(images)
    #是否原图进行了normalization
    #交换之后，(H,W,C)
    image=image.permute(1,2,0)
    image=(image*torch.tensor(std)+torch.tensor(mean))
    #交换之后,(C,H,W)
    image=image.permute(2,0,1)
    #将tensor转化为Image格式
    image=transforms.ToPILImage()(image)
    print(np.array(image))
    #存储图片
    image.save(path)





def tensor2pic2(x,path,dpi=300):
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
    # 去除图像周围的白边
    height, width, channels = im.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / dpi, height / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    plt.savefig(path, dpi=dpi)