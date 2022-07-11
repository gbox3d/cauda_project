#%%
import os
import torch
# import torchvision
import cv2

import time
#import PIL.Image as Image
#from IPython.display import display
import numpy as np

import datetime;

import json

# Setup detectron2 logger
import detectron2


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,GenericMask
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# import argparse
import yaml

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

#%%
with open('./cmd.yaml', 'r') as f:
    cmdConfig = yaml.load(f,Loader=yaml.FullLoader)['evaluate']

print(cmdConfig)



# print(f'detectron : {detectron2.__version__}')
#%%
print(f' {datetime.datetime.now()} torch : {torch.__version__} cuda : {torch.cuda.is_available()}  cv version : {cv2.__version__}' )
# base_path = './weights/epoch_20k_ds8k'
# _anno_file = os.path.join(base_path,'test.json')
base_path = cmdConfig['base_path']
_anno_file = cmdConfig['anno_file']
_image_path = cmdConfig['image_path']
_result_image_dir = cmdConfig['result_image_dir']

#%%
# parser = argparse.ArgumentParser(description="argument parser sample")
# parser.add_argument('--base-dir', type=str)
# parser.add_argument('--anno-file', type=str)
# _args = parser.parse_args()

# base_path = _args.base_dir
# # test_file = _args.
# _anno_file = os.path.join(base_path,_args.anno_file)

#%%
# _result_image_dir = os.path.join(base_path ,'result_images')

# if not os.path.isdir(_result_image_dir) :
#     os.mkdir(_result_image_dir)
#     # print('make dir : ',_result_image_dir)
# else :
#     # print('skip make dir : ',_result_image_dir)
#     pass


#%%


register_coco_instances(
    f"test_set",
    {},
    # os.path.join(dataset_path,dataset_name,d,anno_file_name),
    _anno_file,
    image_root=_image_path
    )

test_dataset_dicts = DatasetCatalog.get(f"test_set")
# _meta_data = get_uclidformat_metadata(_anno_file)
print(f' {datetime.datetime.now()} :  total test images : {len(test_dataset_dicts)}')

#%% 
# setup config data
cfg_instance_seg = get_cfg()
cfg_instance_seg.merge_from_file(os.path.join(base_path,'config.yaml'))
cfg_instance_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_instance_seg.MODEL.WEIGHTS =  os.path.join(base_path, 'model_final.pth') 

# instance segmentation predictor
instance_segmentation_predictor = DefaultPredictor(cfg_instance_seg)
# print('setup predictor')
#%%
_d = test_dataset_dicts[0]
img = cv2.imread( _d["file_name"])
ground_truth_ano = _d["annotations"]

outputs = instance_segmentation_predictor(img)
pred_masks = outputs["instances"].pred_masks.cpu().numpy()
pred_classes = outputs["instances"].pred_classes.cpu().numpy()

#%%
def get_mask_iou(pred_mask,_gt_anno,pred_class):
    gt_mask = np.zeros( img.shape[0:2] ,np.uint8)
    np_cnt =  np.array(_gt_anno["segmentation"],dtype=np.int32).reshape((-1, 2))
    gt_mask = cv2.fillPoly(gt_mask, [np_cnt],(1))

    
    iou = 0
    _mask = ((pred_mask * gt_mask) > 0) # intersection area
    _uinon = ((gt_mask + pred_mask) > 0) # union area
    if _mask.sum() > 0 and pred_class == _gt_anno["category_id"] :
        iou = _mask.sum() / _uinon.sum() 
        # print(f'{i} : {iou}')
        # display(Image.fromarray( _mask))
    
    return  iou
   
#%%
iou_sum = 0
_count = 0
f1_sum = 0
sum_tp = 0
sum_fp = 0
sum_fn = 0


for _d in test_dataset_dicts :
    img = cv2.imread( _d["file_name"])
    ground_truth_ano = _d["annotations"]
    outputs = instance_segmentation_predictor(img)
    pred_masks = outputs["instances"].pred_masks.cpu().numpy()
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    # print(f'image : {_d["file_name"]}')

    _temp = 0
    for i,pred_mask in enumerate(pred_masks) :
        pred_class = pred_classes[i]
        # _found = False
        _maxiou = 0
        for _gt_anno in ground_truth_ano :
            iou = get_mask_iou(pred_mask,_gt_anno,pred_class)
            if iou > _maxiou :
                _maxiou = iou
        iou_sum += _maxiou
        _temp += _maxiou
        _count += 1
    print( f'  {_d["file_name"]} miou {_temp / len(pred_classes)}')

    tp = 0
    fp = 0

    for i,pred_mask in enumerate(pred_masks) :
        _found = False
        pred_class = pred_classes[i]
        for _gt_anno in ground_truth_ano :
            iou = get_mask_iou(pred_mask,_gt_anno,pred_class)
            # print( f" {i} : {iou}" )
            if iou > 0.5 :
                tp += 1
                _found = True
                break;
        if not _found :
            fp += 1

    fn = 0
    for _gt_anno in ground_truth_ano :
        _found = False
        for i,pred_mask in enumerate(pred_masks) :
            pred_class = pred_classes[i]
            iou = get_mask_iou(pred_mask,_gt_anno,pred_class)
            if iou > 0.5 :
                _found = True
                break;
        if not _found :
            fn += 1

    print(f'{datetime.datetime.now()} : TP : {tp}, FP : {fp}, FN : {fn},TN : -')
    if tp > 0 :

        print(f'{datetime.datetime.now()} :  precision : {tp/(tp+fp)} recall : {tp/(tp+fn)} f1 : {2*tp/(2*tp+fp+fn)} accuracy : {(tp)/(tp+fp+fn)}')
        f1_sum += 2*tp/(2*tp+fp+fn)
        
    else :
        print(f'zero accuracy !')
    
    sum_tp += tp
    sum_fp += fp
    sum_fn += fn


print( f'{datetime.datetime.now()} : total detection count { _count } , miou { iou_sum / _count } , f1 { f1_sum / _count }')
print( f'{datetime.datetime.now()} : total => tp {sum_tp} fp {sum_fp} fn {sum_fn}')


# %%
