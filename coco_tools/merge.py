#%%
import os
import random
from unicodedata import category
import cv2
import numpy as np
# from numpy.lib.function_base import append
import torch
import torchvision
import json
import requests
import datetime
import yaml

# import detectron2
from detectron2.structures import BoxMode

#%%

with open('./cmd.yaml', 'r') as f:
    cmdConfig = yaml.load(f,Loader=yaml.FullLoader)['merge']

json_files = cmdConfig['list']
save_name = cmdConfig['save_name']
    

# json_files = [
#     '../temp/dic_1004/anno.json',
#     '../temp/dic_1009/anno.json',
# ]

# %%
annotation = []
images = []
meta_datas = []
categories = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        _categories = data['categories']
        images.extend(data['images'])
        meta_datas.extend(data['meta'])
        
        if(len(categories) == 0):
            for cat in _categories: 
                cat['name'] = cat['supercategory'] + '_' + cat['name']
                categories.append(cat)
            annotation.extend(data['annotations'])
            
        else:
            last_id = categories[-1]['id']
            # print(f'last_id: {last_id}')
            for cat in _categories: 
                cat['id'] += last_id
                cat['name'] = cat['supercategory'] + '_' + cat['name']
                categories.append(cat)
            
            for anno in data['annotations']:
                anno['category_id'] += last_id
                annotation.append(anno)
                
                # get last index


for _cat in categories:
    print(_cat)
        
        
#         annotation.extend(data['annotations'])
#         images.extend(data['images'])
#         meta_datas.extend(data['meta'])


# print( f'len(annotation): {len(annotation)}')
# print( f'len(images): {len(images)}')
# print( f'len(meta_datas): {len(meta_datas)}')

#%%
dataset_dicts = {
        "info": {
            "description": "daisy ai solution",
            "url": "",
            "version": "1",
            "date_created": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        "images": images,
        "annotations": annotation,
        "meta": meta_datas,
        "categories": categories,
    }

# %%
# save_name = '../temp/_data.json'
with open(save_name, 'w') as f:
    json.dump( 
        obj=dataset_dicts, 
        fp=f,
        indent=2 # 줄맞추기
)

print(f'{save_name} 저장 완료 , image num {len(dataset_dicts["images"])}')

# %%
