# cauda_project
Aphid homology analysis AI development project with mask rcnn 

## 1. convert coco
pascal voc 포멧을 coco 포멧으로 변환하기  
```sh
python ./coco_tools/voc2coco.py -d=./dataset/dic_1009 -n=dic_1009 -o=./dataset/dic_1009/anno.json -i ./dataset/images
python ./coco_tools/voc2coco.py -d=./dataset/dic_1004 -n=dic_1004 -o=./dataset/dic_1004/anno.json -i ./dataset/images
```
## 2. merge coco files

```sh
python ./coco_tools/coco_merge.py -c ./settings/cmd.yaml
```
## 3. split coco files

```sh
python ./coco_tools/coco_spliter.py --img-path=./dataset/images --json-path=./dataset/all.json --output-path=./dataset  --train-ratio=0.8

```
## 4. config 파일 만들기 


```sh
python make_cfg.py -s ./settings/cauda.yaml
``` 
## train

```sh
CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file ./configs/cauda_config.yaml -s ./settings/cauda.yaml

python train_net.py --config-file ./configs/cauda_config.yaml -s ./settings/cauda.yaml --num-gpus 1
```
## 탠서보드 

--logdir 옵션으로 학습결과물이 출력되는 디랙토리를 지정한다.<br>

```sh
tensorboard --logdir /home/ubiqos-ai2/work/visionApp/cauda_project/output/all
```
http://localhost:6006 으로 접속한다.<br>
