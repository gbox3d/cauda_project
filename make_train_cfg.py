#%%
import os
from detectron2 import config
# import random
import yaml
import matplotlib.pyplot as plt
# import cv2
# import numpy as np
from detectron2.utils.logger import setup_logger
# import detectron2
import torch
# import torchvision

# check pytorch installation:
print(torch.__version__, torch.cuda.is_available())
# please manually install torch 1.9 if Colab changes its default version
# assert torch.__version__.startswith("1.9")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

import utils
import cfgArgParser

#%%
dataset_path = "./temp"
datasetname = "dic_1009"
image_root = "/home/ubiqos-ai2/work/datasets/bitles/dic_1009/voc"
base_config_file = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
IMS_PER_BATCH = 2
epoch_num = 1000
eval_period = 200

#%% cmd line args
args = cfgArgParser.cfgArgParser().parse_args()

dataset_path = args.dataset_root
datasetname = args.dataset_name
image_root = args.image_root

base_config_file = args.base_config_file
IMS_PER_BATCH = args.batch
epoch_num = args.epoch
eval_period = args.eval_period

#%%
output_path = os.path.join('./output',datasetname)
config_path = os.path.join('./configs',datasetname)

#%%
utils.loadCocoDataset(
    dataset_path = dataset_path,
    dataset_name = datasetname,
    image_root=image_root
    )

ds_test = DatasetCatalog.get(f"{datasetname}_test")
_meta = MetadataCatalog.get(f"{datasetname}_test") # 메타데이터 추출 
print(_meta.thing_classes)
print(f'class num : {len(_meta.thing_classes)}')
NUM_CLASSES = len(_meta.thing_classes)


#%%
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(base_config_file))
# cfg.DATASETS.TRAIN = (datasetname+'_train',datasetname+'_valid')
cfg.DATASETS.TRAIN = (datasetname+'_train') # 학습용 데이터 설정 valid 데이터는 제외
cfg.DATASETS.TEST = (datasetname+'_test',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config_file)
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = epoch_num
cfg.SOLVER.STEPS = []        # do not decay learning rate

cfg.OUTPUT_DIR = output_path
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

if eval_period > 0:
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.EVAL_PERIOD = 100

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(config_path, exist_ok=True)

config_file = os.path.join(config_path,'config.yaml')
# 설정 파일 덤프 하고 astrophysics.yaml 로 저장 
cfg_file = yaml.safe_load(cfg.dump())
with open(config_file, 'w') as f:
    yaml.dump(cfg_file, f)
print(f'{config_file}  done')

# %%
