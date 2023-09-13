"""
@Env: /anaconda3/python3.10
@Time: 2023/6/18-13:46
@Auth: karlieswift
@File: ddp_train_coco.py
@Desc:
"""
import torch
from attackutils  import myGlobal
import warnings
import sys
from transformers import AutoTokenizer
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from transformers import GPT2TokenizerFast
from attackutils.modeling_vision_encoder_decoder import VisionEncoderDecoderModel
from attackutils.image_processing_vit import ViTImageProcessor
import os
import argparse
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
parser = argparse.ArgumentParser()
parser.add_argument("-start", "--start", type=int, default="0")
parser.add_argument("-end", "--end",  type=int, default="6")
args = parser.parse_args()

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        encoding = {}
        encoding["text"] = tokenizer(item["text"], max_length=128, padding="max_length",
                                     return_tensors="pt", ).input_ids

        encoding["image"] = image_processor(item["image"], return_tensors="pt").pixel_values
        return encoding


def collator(batch):
    new_batch = {"images": [], "texts": []}
    for item in batch:
        new_batch["images"].append(item["image"])
        new_batch["texts"].append(item["text"])
    new_batch["images"] = torch.concat(new_batch["images"], dim=0)
    new_batch["texts"] = torch.concat(new_batch["texts"], dim=0)
    return new_batch
class ProgressBar:
    def __init__(self, width=30):
        self.width = width

    def update(self, max_value, current_value, info):
        progress = int(round(self.width * current_value / max_value))
        bar = '=' * progress + '.' * (self.width - progress)
        prefix = '{}/{}'.format(current_value, max_value)

        prefix_max_len = len('{}/{}'.format(max_value, max_value))
        buffer = ' ' * (prefix_max_len - len(prefix))

        sys.stdout.write('\r {} {} [{}] - {}'.format(prefix, buffer, bar, info))
        sys.stdout.flush()

    def new_line(self):
        print()


from datasets import load_dataset

# 
#ds = load_dataset("imagefolder", data_dir='/public/home/mswanghao/TorchProject/vitgpt2/train2017', split="train")
ds = load_dataset('/public/home/mswanghao/.cache/huggingface/datasets/imagefolder/default-677808fe757045b0/0.0.0/37fbb85cc714a338bea574ac6c7d0b5be5aff46c1862c1989b20e0771199e93f',  split="train")
print(ds)

train_ds = ds

print(train_ds[0])

myGlobal._init()


device = "cuda" if torch.cuda.is_available() else "cpu"

# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

lr=0.00005
#lr=0.0001
top_n = 6
start=args.start
end=args.end
myGlobal.set_value('top_n', top_n)
# myGlobal.set_value('soft_max_index', [])
name = 'DRSL3'
print("top_n:", top_n)
if name == 'CE':
    print("CE lr={}".format(lr))
    myGlobal.set_value('b', 0)
    myGlobal.set_value('loss_name', 'CE')
    pt_save_directory = "./vit_gpt2_coco_model3/model_CE_addsoftmax1"

elif name == "DRSL":
    b = 1e-8
    print("DRSL b={} lr={}".format(b,lr))
    myGlobal.set_value('b', b)
    myGlobal.set_value('loss_name', 'DRSL')
    pt_save_directory = "./vit_gpt2_coco_model1/model_DRSL"

elif name == "DRSL1":
    b = 0.00001
    print("DRSL1 b={} top_n={} lr={}".format(b,top_n,lr))
    myGlobal.set_value('b', b)
    myGlobal.set_value('loss_name', 'DRSL1')
    pt_save_directory = "./vit_gpt2_coco_model1/model_DRSL1_{}".format(top_n)
elif name == "DRSL3":
    b = 0.00001
    myGlobal.set_value('b', b)
    myGlobal.set_value('loss_name', 'DRSL3')
    myGlobal.set_value('start',start)
    myGlobal.set_value('end',end)
    print("DRSL1 b={} lr={} start={} end={}".format(b,lr,start,end))
    pt_save_directory = "./vit_gpt2_coco_model3/model_DRSL311111_{}_{}".format(start,end)

if os.path.exists(pt_save_directory):
    model = VisionEncoderDecoderModel.from_pretrained(pt_save_directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(pt_save_directory)
    image_processor = ViTImageProcessor.from_pretrained(pt_save_directory)
else:
    image_encoder_model = "google/vit-base-patch16-224-in21k"
    text_decode_model = "/public/home/mswanghao/TorchProject/vitgpt2/gpt2checkpont"
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    image_processor = ViTImageProcessor.from_pretrained(image_encoder_model)
model.to(device)
tokenizer.pad_token = tokenizer.eos_token
# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

output_dir = "vit-gpt-model-base"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
image_processor.save_pretrained(output_dir)

MAX_PATCHES = 1024


env_dict = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
}
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
rank = int(os.environ["RANK"])
print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
torch.cuda.set_device(local_rank)

train_dataset = ImageCaptioningDataset(train_ds)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8, collate_fn=collator, sampler=train_sampler)

print("训练前")
model.eval()
pokemon = train_ds[0]['image']


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
orgin_image = None
progress_Bar = ProgressBar()
'''
"./vit_gpt2_coco_model/model1":CE
"./vit_gpt2_coco_model/model2":原来的DRSL-非目标
"./vit_gpt2_coco_model/model3":现在的DRSL-最大值
'''


model = model.cuda(local_rank)
ddp_model = DDP(model, [local_rank], find_unused_parameters=True)
ddp_model.train()

for epoch in range(1,3):

    train_dataloader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(train_dataloader):

        pixel_values = batch['images'].to(local_rank)
        labels = batch['texts'].to(local_rank)

        loss = ddp_model(pixel_values=pixel_values, labels=labels).loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_Bar.update(max_value=len(train_dataloader), current_value=idx + 1,
                            info="当前Epoch{}的batch的loss:{:.4f}".format(1, loss.item()))
        if idx % 100 == 0 and idx > 1:
            model.eval()
            dog = Image.open('images/dog.jpg')
            pixel_values = image_processor(dog, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_text1)
            dog = Image.open('images/dog_and_cat.jpg')
            pixel_values = image_processor(dog, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            generated_text1 = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(generated_text1)
            image_processor.save_pretrained(pt_save_directory)
            model.save_pretrained(pt_save_directory)
            tokenizer.save_pretrained(pt_save_directory)

            model.train()
            progress_Bar.new_line()
dist.destroy_process_group()

# export MASTER_ADDR='127.0.0.1'
# export MASTER_PORT='29500'
# #export LOGLEVEL="DEBUG"
# torchrun \
#     --nnodes=1:3\
#     --nproc_per_node=4\
#     --max_restarts=3\
#     --rdzv_id=1\
#     --rdzv_backend=c10d\
#     --rdzv_endpoint='127.0.0.1'\
#      ddp_train_coco.py
