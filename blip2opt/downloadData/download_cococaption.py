from blip2opt.datasets.builders import load_dataset
print(111)
coco_dataset = load_dataset("coco_caption")

print(coco_dataset.keys())
# dict_keys(['train', 'val', 'test'])

print(len(coco_dataset["train"]))
# 566747

print(coco_dataset["train"][0])
