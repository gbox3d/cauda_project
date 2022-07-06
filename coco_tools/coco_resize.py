#%%
import os
import cv2
import numpy as np
import json

from detectron2.structures import BoxMode
import PIL.Image as Image
from IPython.display import display

import argparse

#%%
json_file = './output/test.json'
img_root = '../../datasets/mushroom_data/yangsongyi/_image/images'
output_path = './output'
resize_ratio = 0.5

#%%
parser = argparse.ArgumentParser(description="argument parser sample")

parser.add_argument('--json-file', type=str)
parser.add_argument('--img-root', type=str)
parser.add_argument('--output-path', type=str)
parser.add_argument('--resize-ratio',default=0.5, type=float)

_args = parser.parse_args()

json_file = _args.json_file
img_root = _args.img_root
output_path = _args.output_path
resize_ratio = _args.resize_ratio

#%%

with open(json_file) as f:
    dataObj = json.load(f)

    _dataObj = dataObj.copy()

    _img_sets = _dataObj['images']

    _metaData = list({v['id']:v for v in _dataObj['meta']}.values())
    _dataObj['annotations'] = [ _anno for _anno in _dataObj['annotations'] if _anno['image_id'] in [_img['id'] for _img in _img_sets] ]
    _dataObj['meta'] = [ _meta for _meta in _metaData if _meta['id'] in [_img['meta_id'] for _img in _img_sets] ]



    for index,imgset in enumerate(_img_sets):
        if  '/' in imgset['file_name'] :
            file_name = imgset['file_name'].split('/')[1]
            _img_id = imgset['id']
            imgset['file_name'] = file_name
        
        # print(_img_set[0])
        img_file = os.path.join(img_root,file_name) 
        np_img = cv2.imread(img_file)
        np_img = cv2.resize(np_img,dsize=(0,0),fx=resize_ratio,fy=resize_ratio,interpolation=cv2.INTER_AREA) # 이미지 사이즈 50% 조정
        
        # display(Image.fromarray( cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)))

        encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
        _,_encodee_img = cv2.imencode('.jpg',np_img,encode_param)

        _annos = [_anno for _anno in _dataObj['annotations'] if _anno['image_id'] == _img_id]

        for _anno in _annos:
            _anno['width'] = np_img.shape[1]
            _anno['height'] = np_img.shape[0]
            # print(np.array(_anno['segmentation']).ndim)
            _poly = np.array(_anno['segmentation'],dtype=np.int32).flatten()/2
            _anno['segmentation'] = [(_poly).tolist()]
            # print(_anno)

            _poly = np.array(_anno['segmentation'],dtype=np.int32).flatten()
            np_cnt = _poly.reshape(-1,2)
            x,y,w,h = cv2.boundingRect(np_cnt)
            _anno['bbox'] = [x,y,w,h]

            if _anno['iscrowd'] == 'Y':
                _anno['iscrowd'] = 0

        with open(os.path.join(output_path,'resized',file_name),'wb') as fd :
            fd.write( _encodee_img.tobytes() )
        # print(f'{file_name} saved')
        print(f' [{int(index/len(_img_sets)*100)}%] {file_name} saved  ',end='\r')
    
    save_name = os.path.join(output_path,'coco_annotation.json')
    with open(save_name, 'w') as f:
        anno_num = len(_dataObj["annotations"])
        print(f'{save_name} saved annotaion  , anno total :{anno_num} , img total : img {len(_img_sets)}')
        json.dump(_dataObj, fp=f,indent=2)
    