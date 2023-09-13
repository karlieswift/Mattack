import pandas as pd
from tqdm import tqdm
import os
import shutil
df=pd.read_csv('/public/home/mswanghao/TorchProject/vitgpt2/train2017/metadata.csv')

for i in tqdm(range(df.shape[0])):
    image=df.iloc[i,0]
    image=os.path.join('/public/home/mswanghao/TorchProject/vitgpt2/train2017',image)
    shutil.copy(image,'./newcoco2017')
