#%%
import cv2
import os
import numpy as np
import json
import requests
import io

from PIL import Image
from IPython.display import display
print(f'cv version : {cv2.__version__}')
#%%
with open('./temp/output_segmentation.json') as json_file:
    json_data = json.load(json_file)
    print(json_data['categories'])
    # images,annotations, categories
# _jsonObj = json.loads('./temp/output_segmentation.json')

# print(_jsonObj['info'])
# %%
print(json_data['info'])
print(len(json_data['images']))
output_dir = '../../datasets/car_street/img'

# %% 이미지 다운받기 
for image_info in json_data['images'] :
    # print(file['coco_url'])
    _url = image_info['coco_url']
    _cmd = f'wget {_url} -P {output_dir}  '
    os.system(_cmd)
    # print('done')

#%%
url = json_data["images"][0]['coco_url']
file_name = json_data["images"][0]['file_name']
_id = json_data["images"][0]['id']

response = requests.get(url)
buf = np.ndarray(
    shape=(1, len(response.content)), 
    dtype=np.uint8, 
    buffer=response.content
    )
_img = cv2.imdecode(buf,cv2.IMREAD_COLOR)
print(_img.shape)

image_json = {
    'height' : _img.shape[0],
    'width' : _img.shape[1],
    'file_name' : file_name,
    'id' : _id
}
print(image_json)
json.dumps(image_json)


# print(response.headers)
# print(response.status_code)
file_path = output_dir + file_name
if os.path.isfile(file_path) == False :
    with open(file_path,"wb") as img_file :
        img_file.write(response.content)
        print(f'save {file_name}')
else :
    print(f'skip {file_name}')
# print(file_name)
# display(Image.fromarray(cv2.cvtColor(_img,cv2.COLOR_BGR2RGB)))


# %%


a = []
file_path = f'{output_dir}/{json_data["images"][0]["file_name"]}'
img = cv2.imread(file_path)
print(img.shape)
image_json = {
    'height' : img.shape[0],
    'width' : img.shape[1],
    'file_name' : json_data["images"][0]["file_name"]
}

print(image_json)
json.dumps(image_json)

a.append(image_json)
print(a)


# display(Image.fromarray( cv2.cvtColor(img,cv2.COLOR_BGR2RGB) ))


# %%
