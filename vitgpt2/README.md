# vitgpt2

This repository contains the code and data for the following paper:

[DRSL: Distribution-restrained Softmax Loss for the Safety of Large Language Models](https://arxiv.org/)

```
@inproceedings{ 
  title={DRSL: Distribution-restrained Softmax Loss for the Safety of Large Language Models},
}
```

## Requirements

- Python 3 (tested with Python 3.9)
- Install required packages:

```bash
python -m pip install -r requirements.txt
```

## VITGPT2 Experiments

### Preparation

#### Prepare data

 COCO2017   |                                               source                                                
------------|:---------------------------------------------------------------------------------------------------:
 images     |               <a href="http://images.cocodataset.org/zips/train2017.zip">Download</a>               
 annotation | <a href="http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip">Download</a> 

After the data download is completed, extract it to the specified folder, and import the data using the datasets
library. Therefore, it is necessary to prepare the data in the format of datasets, as well as prepare the metadata.csv
file and place it in the train2017 folder. The content format of metadata.csv is:

 file_name        |                           text                           
------------------|:--------------------------------------------------------:
 000000391895.jpg | A man with a red helmet on a small moped on a dirt road. 
 000000522418.jpg |    A woman wearing a net on her head cutting a cake.     

a detailed tutorials on datasets can be found in [datasets](https://huggingface.co/docs/datasets/index).

Then:

```
cd datasets
python load_coco.py
```



### Training

**training cross entropy lossfunction**  
Need to change the name of the code file ddp_train_coco.py On line 102.  
1.set name='CE'
```bash
torchrun nproc_per_node=4 ddp_train_coco.py
```

**training DRSL lossfunction**  
Need to change the name of the code file ddp_train_coco.py    
1.on line 97.set top_n=n   
2.on line 102.set name='DRSL3'
```bash
torchrun nproc_per_node=4 ddp_train_coco.py -start m -end n
```


### Attack
```bash
python adv_vitgpt2.py
```


### Attack results
![find wild](attacked.jpg)  


 
