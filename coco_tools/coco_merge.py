#%%
import os
import random
import cv2
import numpy as np
# from numpy.lib.function_base import append
import torch
import torchvision
import json
import requests
# import detectron2
from detectron2.structures import BoxMode

#%%
json_files = [
    '/home/gbox3d/work/datasets/mushroom_data/yangsongyi/data.json',
    '/home/gbox3d/work/datasets/mushroom_data/yangsongyi_v2/data.json',
    '/home/gbox3d/work/datasets/mushroom_data/yangsongy1_v3/_label/data.json',
]


# %%
annotation = []
images = []
meta_datas = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        annotation.extend(data['annotations'])
        images.extend(data['images'])
        meta_datas.extend(data['meta'])


print( f'len(annotation): {len(annotation)}')
print( f'len(images): {len(images)}')
print( f'len(meta_datas): {len(meta_datas)}')

#%%
with open(json_files[0]) as f:
    data = json.load(f)
    __dataObj = data.copy()
    __dataObj['annotations'] = annotation
    __dataObj['images'] = images
    __dataObj['meta'] = meta_datas

# %%
save_name = './_data.json'
with open(save_name, 'w') as f:
    json.dump(__dataObj, f)

print(f'{save_name} 저장 완료 , image num {len(__dataObj["images"])}')

# %%
