#%%
import os
import string
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
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

#%%
import argparse
parser = argparse.ArgumentParser(description="argument parser sample")

parser.add_argument(
    '-s','--settings', type=str, 
    default='./settings.yaml',
    help='choose  your settings file')

settgins_file = parser.parse_args().settings


#%%
print(f' settings file : {settgins_file}')
with open(settgins_file) as f :
    _config = yaml.load(f, Loader=yaml.FullLoader)
    # print(_config)
    
#%%
register_coco_instances("test_set", {}, 
    _config['test_set']['anno'], 
    _config['test_set']['img_dir'])

#%%
datasetname = _config['dataset']['name']
base_config_file = _config['cfg']['base_config_file']
output_dir = _config['cfg']['output_dir']
config_dir = _config['cfg']['config_dir']


MAX_ITER = _config['cfg']['max_iter']
IMS_PER_BATCH = _config['cfg']['ims_per_batch']

# config_name = 'config.yaml'
config_path_out = os.path.join(_config['cfg']['config_dir'], f"{datasetname}_config.yaml")

#%%
# get count of classses
ds_test = DatasetCatalog.get(f"test_set")
_meta = MetadataCatalog.get(f"test_set") # 메타데이터 추출 
print(_meta.thing_classes)
print(f'class num : {len(_meta.thing_classes)}')
NUM_CLASSES = len(_meta.thing_classes)

#%%
os.makedirs(output_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)


#%%
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(base_config_file))
cfg.DATASETS.TRAIN = ('train',)
cfg.DATASETS.TEST = ('test',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config_file)
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = []        # do not decay learning rate

cfg.OUTPUT_DIR = output_dir
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

cfg.TEST.AUG.ENABLED = True
cfg.TEST.EVAL_PERIOD = _config['cfg']['eval_period']    


# 설정 파일 덤프 하고 astrophysics.yaml 로 저장 
cfg_file = yaml.safe_load(cfg.dump())
with open(config_path_out, 'w') as f:
    yaml.dump(cfg_file, f)

print(f'{config_path_out}  done')