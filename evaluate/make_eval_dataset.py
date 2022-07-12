#%%
import os
from unicodedata import category
import cv2
import numpy as np

#import the COCO Evaluator to use the COCO Metrics
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.structures import BoxMode

#%%
weight_path = '../output/all'
#register your data
# register_coco_instances("my_dataset_train", {}, "/code/detectron2/detectron2/instances_train2017.json", "/code/detectron2/detectron2/train2017")
# register_coco_instances("my_dataset_val", {}, "/code/detectron2/detectron2/instances_val2017.json", "/code/detectron2/detectron2/val2017")
register_coco_instances("my_dataset_test", {}, 
                        json_file = '/home/ubiqos-ai2/work/visionApp/cauda_project/temp/all/test.json',
                        image_root = '../../../datasets/bitles/images')
                        # "/code/detectron2/detectron2/instances_test2017.json", 
                        # "/code/detectron2/detectron2/test2017")

#load the config file, configure the threshold value, load weights 
# cfg = get_cfg()
# cfg.merge_from_file("/code/detectron2/detectron2/output/custom_mask_rcnn_X_101_32x8d_FPN_3x_Iteration_3_dataset.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# cfg.MODEL.WEIGHTS = "/code/detectron2/detectron2/output/model_final.pth"

test_dataset_dicts = DatasetCatalog.get(f"my_dataset_test")
# print(f' {datetime.datetime.now()} :  total test images : {len(test_dataset_dicts)}')

#%%
cfg_instance_seg = get_cfg()
cfg_instance_seg.merge_from_file(os.path.join(weight_path,'config.yaml'))
cfg_instance_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_instance_seg.MODEL.WEIGHTS =  os.path.join(weight_path,'model_final.pth') #f'./output/{dataset_name}/model_final.pth'

cfg = cfg_instance_seg
# Create predictor
predictor = DefaultPredictor(cfg)

#%%
_meta = MetadataCatalog.get(f"my_dataset_test")
thing_classes = _meta.thing_classes
thing_dataset_id_to_contiguous_id = _meta.thing_dataset_id_to_contiguous_id

for _d in test_dataset_dicts :
    img = cv2.imread( _d["file_name"])
    
    # gt_class = _d['category_id']
    outputs = predictor(img)
    pred_boxes = np.array([box.cpu().numpy() for box in outputs["instances"].pred_boxes])
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()
    
    _x =  pred_boxes[:,0]
    _y = pred_boxes[:,1]
    _right = pred_boxes[:,2]
    _bottom = pred_boxes[:,3]
    
    with open(f'./detection_result/{_d["image_id"]}.txt', 'w') as f:
        for i in range(len(pred_boxes)):
            f.write(f'{ thing_classes[pred_classes[i]]} {pred_scores[i]} {round(_x[i])} {round(_y[i])} {round(_right[i])} {round(_bottom[i])} \n')
            
    # ground truth
    ground_truth_ano = _d["annotations"]
    
    with open(f'./ground_truth/{_d["image_id"]}.txt', 'w') as f:
        for i in range(len(ground_truth_ano)):
            _gt = ground_truth_ano[i]
            if _gt['bbox_mode'] == BoxMode.XYWH_ABS:
                x = _gt['bbox'][0]
                y = _gt['bbox'][1]
                _right = _gt['bbox'][2] + x
                _bottom = _gt['bbox'][3] + y
            # elif ground_truth_ano['bbox_mode'] == BoxMode.XYWH_REL:
            #     x = ground_truth_ano['bbox'][0] * img.shape[1]
            #     y = ground_truth_ano['bbox'][1] * img.shape[0]
            #     _right = ground_truth_ano['bbox'][2] * img.shape[1] + x
            #     _bottom = ground_truth_ano['bbox'][3] * img.shape[0] + y
            elif _gt['bbox_mode'] == BoxMode.XYXY_ABS:
                x = _gt['bbox'][0]
                y = _gt['bbox'][1]
                _right = _gt['bbox'][2]
                _bottom = _gt['bbox'][3]
            
            f.write(f'{ thing_classes[_gt["category_id"]]} {round(x)} {round(y)} {round(_right)} {round(_bottom)} \n')
    
    
# %%
