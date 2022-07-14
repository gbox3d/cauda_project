#%%
import os
from posixpath import dirname
import numpy as np
import xml.etree.ElementTree as ET
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

def _parseXml(annotation_path,image_path,filename,classes) :

    imgObj = {}
    root = ET.parse(os.path.join(annotation_path, filename)).getroot()
    imgObj["file_name"] = root.find('filename').text
    _img = cv2.imread(
        os.path.join(image_path, root.find('filename').text)
        )
    imgObj["height"] = _img.shape[1]
    imgObj["width"] = _img.shape[0]
    imgObj["id"] = uuid.uuid3(uuid.NAMESPACE_DNS, filename).int>>64

    metaObj = {
        "id": uuid.uuid1().int>>64,
    }
    imgObj['meta_id'] = metaObj['id']

    
    # print(imgObj["image_id"])
    # record["annotations"] = _parseXml(root,classes)
    anno_objs = []
    for member in root.findall('object'):
        # print(member[4])
        _label = member.find('name').text
        # print(_label)
        _bbox = member.find('bndbox')
        # _seg = member.find('segmentation')
        obj = {}
        if _bbox :
            xmin = float(_bbox.find('xmin').text)
            ymin = float(_bbox.find('ymin').text)
            xmax = float(_bbox.find('xmax').text)
            ymax = float(_bbox.find('ymax').text)
            
            # xmax =  float(_bbox.find('xmax').text)
            # ymax =  float(_bbox.find('ymax').text)
            
            obj = {
                'bbox': [ round(xmin), round(ymin),round(xmax), round(ymax)],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': classes.index(_label),
                "iscrowd": 0,
                "image_id": imgObj["id"]
            }
            anno_objs.append(obj)
        else:
            segment = member.findall('segmentation')
            
            _poly= np.array([ [ round(float(_seg.find('x').text)),round(float(_seg.find('y').text))] for _seg in segment])

            xmin = np.min(_poly[:,0])
            ymin = np.min(_poly[:,1])
            xmax = np.max(_poly[:,0])
            ymax = np.max(_poly[:,1])

            # for _seg in segment :
            #     _x = float(_seg.find('x').text)
            #     _y = float(_seg.find('y').text)
            _poly = _poly.flatten().tolist()
            obj = {
                'bbox': [ round(xmin),round( ymin), round(xmax), round(ymax)],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': classes.index(_label),
                "iscrowd": 0,
                "segmentation": [ _poly],
                "image_id": imgObj["id"]
            }
            anno_objs.append(obj)
        obj['area'] = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
        obj['id'] = uuid.uuid1().int>>64
    # record["annotations"] = anno_objs
    return {
        'image': imgObj,
        'annotations': anno_objs,
        'meta': metaObj
    }
#%%
def loadVocDataset(annotation_path,image_path,classes,superset=None) :

    #annotation_path =  os.path.join(dataset_path,dataset_name,'voc')
    # image_path = annotation_path

    dataset_dicts = {
        "info": {
            "description": "daisy ai solution",
            "url": "",
            "version": "1.1",
            "date_created": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        "images": [],
        "annotations": [],
        "meta":[],
        "categories": [ {'id' :idx, 'name' : name , 'supercategory' : superset } for idx,name in enumerate(classes) if name != '__background__'],
    }
    for idx, filename in enumerate(os.listdir(annotation_path)):
        # print(filename)
        _filename, file_extension = os.path.splitext(filename)
        if file_extension.lower() == '.xml' :
            vocData = _parseXml(annotation_path,image_path,filename,classes)
            dataset_dicts["images"].append(vocData["image"])
            dataset_dicts["annotations"].extend(vocData["annotations"])
            dataset_dicts["meta"].append(vocData["meta"])
            
    

    return dataset_dicts
#%%
dataset_path= "/home/ubiqos-ai2/work/datasets/bitles"
dataset_name='dic_1009'
save_name='./temp/anno.json'

#%%
import argparse
parser = argparse.ArgumentParser(description="voc to coco converter")
parser.add_argument('--dataset-path','-d', type=str, help='help : data set path')
parser.add_argument('--dataset-name','-n', type=str, help='help : data set name')
parser.add_argument('--img-path','-i', type=str,default=None,help='help : image path')
parser.add_argument('--save-name','-o', type=str,help='help : output name')


_args = parser.parse_args()

dataset_name = _args.dataset_name
dataset_path = _args.dataset_path
save_name = _args.save_name
img_path = _args.img_path
if img_path is None :
    img_path = os.path.join(dataset_path,dataset_name,'voc')

#%% load config data
# print(f'load config data from {dataset_path}')
config_data = {}
label_dic = {}
dataset_config_file = os.path.join(dataset_path,dataset_name,'dataset_info.yaml')

# print(dataset_config_file)

try:
    with open(dataset_config_file, 'r') as f:
        config_data = yaml.safe_load(f)
        # for _index,_lb in enumerate(config_data['class_names'] ):
        #     label_dic[_lb] = _index
    print(f'complete load config data from {dataset_config_file}')
except  Exception as ex:
    print('error : ')   
    config_data = None 
    print(ex)
config_data['class_names'].insert(0,'__background__')


#%%

_cocoData = loadVocDataset(
    annotation_path=os.path.join(dataset_path,dataset_name,'voc'),
    image_path= img_path,
    classes=config_data['class_names'],
    superset=dataset_name
    )


#%%
with open(save_name, 'w') as f:
    json.dump(
        obj=_cocoData, 
        fp=f,
        indent=2 # 줄맞추기
    )

print(f'{save_name} 저장 완료 , image num {len(_cocoData["images"])} , annotation num {len(_cocoData["annotations"])}')
