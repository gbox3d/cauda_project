#%%
import os
import torch
# import torchvision
import cv2
print(f'torch : {torch.__version__}' )
print(f'cuda : {torch.cuda.is_available()}')
print(f'cv version : {cv2.__version__}')

import random
import time
import PIL.Image as Image
from IPython.display import display
import numpy as np

import matplotlib.pyplot as plt

# Setup detectron2 logger
import detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,GenericMask
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.data.datasets import register_coco_instances

print(f'detectron : {detectron2.__version__}')
#%%
# dataset_path = './temp'
dataset_name = 'all'
# annofile = 'train.json'


# print(f'annofile : {annofile} dataset_name : {dataset_name} dataset_path : {dataset_path}')

# _anno_file = os.path.join(dataset_path,dataset_name,test_set_annofile)

"""
Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
"""

register_coco_instances(
    name = f"{dataset_name}",
    metadata = {
        "description": f"hello {dataset_name} dataset",
        },
    json_file = './temp/all/anno.json',
    image_root = '../../datasets/bitles/images'
    )
#%%
dataset_dicts = DatasetCatalog.get(f"{dataset_name}")
fruits_nuts_metadata = MetadataCatalog.get(f"{dataset_name}")

# _meta_data = get_uclidformat_metadata(_anno_file)
print(f'fruits_nuts_metadata : {fruits_nuts_metadata}')
print(f'classes : {fruits_nuts_metadata.thing_classes}')
print(f'description : {fruits_nuts_metadata.description}')


#%%
for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=1)
    vis = visualizer.draw_dataset_dict(d)
    plt.figure()
    plt.imshow(vis.get_image())


# %% pillow 로 출력하기 

_d = dataset_dicts[70]
img = cv2.imread(_d["file_name"])
visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=1)
vis = visualizer.draw_dataset_dict(_d)
display( Image.fromarray( vis.get_image() ))

print(_d)

    
# %%
