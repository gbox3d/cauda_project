#%%
import os
from posixpath import dirname
from unicodedata import category
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree

import torch
import cv2
import time
import datetime
import yaml
import uuid
import json


from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

#%%
coco_file = '/home/ubiqos-ai2/work/visionApp/datasets/test3/fruts_nuts/trainval.json'
output_dir = '/home/ubiqos-ai2/work/visionApp/datasets/test3/fruts_nuts/labels'


#%%
with open(coco_file, 'r') as f:
    coco_json = json.load(f)

print(coco_json['categories'])

#%%
images = coco_json['images']
categories = coco_json['categories']
annos = coco_json['annotations']

#%%
try :
    anno_list = [ {'file_name' : image['file_name'],
                'anno' : [{
                    'segmentation': anno['segmentation'],
                        'iscrowd': anno['iscrowd'],
                        'area': anno['area'],
                        'bbox': anno['bbox'],
                        'category_name': [category for category in coco_json['categories'] if category['id'] == anno['category_id']][0]['name']
                    } for anno in annos if anno['image_id'] == image['id']],
                'width' : image['width'],
                'height' : image['height']} for image in images]
except  Exception as e :
    print(f'error in conver list : {e}')
    exit()
    


#%%
for anno in anno_list:
    _filename = os.path.join(output_dir, f'{ os.path.basename( os.path.splitext( anno["file_name"])[0]) }.json')
    os.makedirs(os.path.dirname(_filename), exist_ok=True) # make dir if not exist
    with open(_filename, "w") as _f:
        json.dump(
            obj=anno, 
            fp=_f,
            indent=2 # 줄맞추기
        )
    print(f'complete save {_filename}')
print('done')
    
    
