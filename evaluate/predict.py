#%%
import os
import torch
# import torchvision
import cv2
print(f'torch : {torch.__version__}' )
print(f'cuda : {torch.cuda.is_available()}')
print(f'cv version : {cv2.__version__}')

import time
import PIL.Image as Image
from IPython.display import display
import numpy as np

import json

# Setup detectron2 logger
import detectron2
import yaml

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,GenericMask
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances

class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJsonEncoder, self).default(obj)

print(f'detectron : {detectron2.__version__}')
#%%
# isNoteBook = True
# weight_path = '/home/ubiqos-ai2/work/visionApp/cauda_project/output/mask_rcnn_X_101_32x8d_FPN_3x'
# image_path = '/home/ubiqos-ai2/work/visionApp/cauda_project/1655858423382.jpg'

# #%%
# isNoteBook = True

# import argparse
# parser = argparse.ArgumentParser(description="coco dataset spliter")
# parser.add_argument('--weights-path','-w', type=str, help='help : wights path')
# parser.add_argument('--image-path','-i', type=str,help='help : image path')
    
# _args = parser.parse_args()
# weight_path = _args.weights_path
# image_path = _args.image_path
# output_path = _args.output

# print(f'wight path : {weight_path}')

with open('./cmd.yaml', 'r') as f:
    cmdConfig = yaml.load(f,Loader=yaml.FullLoader)
    
    isNoteBook = cmdConfig['predict']['isNoteBook']
    weight_path = cmdConfig['predict']['weight_path']
    image_path = cmdConfig['predict']['image_path']
    output_path = cmdConfig['output_path']
    select_device = cmdConfig['device']

    os.makedirs(output_path, exist_ok=True)
    
#%% 
# setup config data
cfg_instance_seg = get_cfg()
cfg_instance_seg.merge_from_file(os.path.join(weight_path,'config.yaml'))
cfg_instance_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_instance_seg.MODEL.WEIGHTS =  os.path.join(weight_path,'model_final.pth') #f'./output/{dataset_name}/model_final.pth'
if select_device == 'auto':
    cfg_instance_seg.MODEL.DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'
else:
    cfg_instance_seg.MODEL.DEVICE = select_device # cuda' if torch.cuda.is_available() else 'cpu'
# instance segmentation predictor
instance_segmentation_predictor = DefaultPredictor(cfg_instance_seg)
print(f'setup predictor {weight_path}')

#%%
_meta = MetadataCatalog.get(cfg_instance_seg.DATASETS.TRAIN[0])
_meta.thing_colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
print(_meta)
# print( f'type : {_meta.evaluator_type}' )
# print(f'image root : {_meta.image_root}')
# print(f'json file path : {_meta.json_file}')
# print(f'class list :  {_meta.thing_classes}')
print(f'name : {_meta.name}')
print(f'name : {_meta.thing_colors}')

#%%
img = cv2.imread(image_path )

start_tick = time.time()
outputs = instance_segmentation_predictor(img)
end_tick = time.time()
print(f'predict time : {end_tick - start_tick}')

pred_masks = outputs["instances"].pred_masks.cpu().numpy()
generic_masks = [GenericMask(x, img.shape[0], img.shape[1]) for x in pred_masks] #마스크데이터를 폴리곤 형태로 변환
pred_classes = outputs["instances"].pred_classes.cpu().numpy()
pred_scores = outputs["instances"].scores.cpu().numpy()
pred_boxes = np.array([box.cpu().numpy() for box in outputs["instances"].pred_boxes])

print(f'found {len(pred_boxes)} objects')

#%%
# print(generic_masks)
_result = [  { 
              'score' : pred_scores[i],
              'class' : pred_classes[i],
              'box' : pred_boxes[i].tolist(),
              'mask' : [ polygon for polygon in gen_mask.polygons ],
              
              }
           for i,gen_mask in enumerate(generic_masks)]

with open(os.path.join(output_path,'predict.json'), 'w') as f:
    json.dump(_result, f,cls=NumpyJsonEncoder)
    
# print(f'output result to {}')
#%% 결과 이미지 출력 
_img = img.copy()
for i,gen_mask in enumerate(generic_masks) :
    for polygon in gen_mask.polygons:
        np_cnt = np.array(polygon,dtype=np.int32).reshape((-1, 2))
        _img = cv2.polylines(_img, [np_cnt], True, (0,0,255), thickness=4) # 새크먼트 그리기 
    
    # print(pred_boxes[i])
    _img = cv2.rectangle(_img, 
                         (int(pred_boxes[i][0]), int(pred_boxes[i][1]) ), # 좌상
                         (int(pred_boxes[i][2]), int(pred_boxes[i][3]) ), # 우하
                         (0,255,0), thickness=4) 


#%%
if isNoteBook:
    __img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    display( Image.fromarray(__img) )
# %%
cv2.imwrite(os.path.join(output_path,'predict.jpg'),_img)

print(f'predict result image saved to {output_path}')
