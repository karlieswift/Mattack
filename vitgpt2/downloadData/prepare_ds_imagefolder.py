from datasets import load_dataset
print(1)
ds = load_dataset("imagefolder", data_dir='/public/home/mswanghao/TorchProject/pix2Struct/newtrain2017', split="train")
print(ds)
