import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

paths = "/public/home/mswanghao/image_caption/natural_images"

for path in os.listdir(paths):
    image_paths=os.path.join(paths,path)
    list_pic=os.listdir(image_paths)
    for image_path in tqdm(list_pic):
        image_path = os.path.join(image_paths, image_path)
        n_dim = len(np.array(Image.open(image_path)).shape)
        if n_dim != 3:
            continue
        shutil.copy(image_path, './natural_images1000')

