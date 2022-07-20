#%%
import os
import random
from unicodedata import category
# import cv2
import numpy as np
# from numpy.lib.function_base import append
# import torch
# import torchvision
import json
# import requests
import datetime
import yaml

# import detectron2
# from detectron2.structures import BoxMode

#%%
# with open('./mergy.yaml', 'r') as f:
#     cmdConfig = yaml.load(f,Loader=yaml.FullLoader)['merge']
# json_files = cmdConfig['list']
# save_name = cmdConfig['save_name']

json_files = [
    '/home/ubiqos/work/visionApp/annos/dic_1011/anno.json',
    '/home/ubiqos/work/visionApp/annos/dic_hongy15/anno.json',
    '/home/ubiqos/work/visionApp/annos/dic_sso3961/anno.json'
]
save_name = './temp/mergy.json'

#%%
import argparse
parser = argparse.ArgumentParser(description="coco mergy")
parser.add_argument('--config','-c', type=str, 
    help='help : config file path, ex> cmd.yaml')

parsed_args = parser.parse_args()

with open(parsed_args.config, 'r') as f:
    cmdConfig = yaml.load(f,Loader=yaml.FullLoader)

json_files = cmdConfig['list']
save_name = cmdConfig['save_name']


# %%
annotation = []
images = []
meta_datas = []
categories = []

for json_file in json_files:
    with open(json_file) as f:
        data = json.load(f)
        _categories = data['categories']

        if(len(categories) == 0):
            images.extend(data['images'])
            meta_datas.extend(data['meta'])
            for cat in _categories: 
                cat['name'] = cat['supercategory'] + '_' + cat['name']
                categories.append(cat)
            annotation.extend(data['annotations'])
            
        else:
            last_cat_id = categories[-1]['id']
            
            can_last = [meta_datas[-1]['id'],images[-1]['id'],annotation[-1]['id']]
            # print(can_last)
            _last_id = max(can_last)
            # print(f'last_id: {last_id}')
            for cat in _categories: 
                cat['id'] += last_cat_id
                cat['name'] = cat['supercategory'] + '_' + cat['name']
                categories.append(cat)
            
            for meta in data['meta']:
                meta['id'] += _last_id
                meta_datas.append(meta)
            
            for img in data['images']:
                img['id'] += _last_id
                img['meta_id'] += _last_id
                images.append(img)
            
            for anno in data['annotations']:
                anno['id'] += _last_id
                anno['category_id'] += last_cat_id
                anno['image_id'] += _last_id
                annotation.append(anno)
                
                # get last index


for _cat in categories:
    print(_cat)
        

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
os.makedirs(os.path.dirname(save_name), exist_ok=True) # make dir if not exist
with open(save_name, 'w') as f:
    json.dump( 
        obj=dataset_dicts, 
        fp=f,
        indent=2 # 줄맞추기
)

print(f'{save_name} 저장 완료 , image num {len(dataset_dicts["images"])} , anno num {len(dataset_dicts["annotations"])}')

# %%
