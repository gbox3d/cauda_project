# cauda_project
Aphid homology analysis AI development project with mask rcnn 

데이터 전처리 과정은 1~3 까지의 과정이다.  
각각의 아이디별로 작업된 데이터셋들을 코코데이터셋으로 변경하고 이것을 통합시킨다. 이때 라벨링값들은 아이디값이 슈퍼셋으로 하고 라벨링명의 접두어로 붙여져서 통합된다.  
마지막으로 훈련세트와 검증세트를 나눈다.  
위과정은 훈련머신보다는 수집 머신에서 행해진다.  
그래서 훈연머신에게 전달하는 식으로 처리가 된다.  

## 1. convert coco
pascal voc 포멧을 coco 포멧으로 변환하기  
```sh
python ./coco_tools/voc2coco.py -d=./dataset/dic_1009 -n=dic_1009 -o=./dataset/dic_1009/anno.json -i ./dataset/images
python ./coco_tools/voc2coco.py -d=./dataset/dic_1004 -n=dic_1004 -o=./dataset/dic_1004/anno.json -i ./dataset/images

python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_hongy15 -n=dic_hongy15 -o=/home/ubiqos/work/dataset/test2/dic_hongy15/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_sso3961 -n=dic_sso3961 -o=/home/ubiqos/work/dataset/test2/dic_sso3961/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_9453 -n=dic_9453 -o=/home/ubiqos/work/dataset/test2/dic_9453/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_1011 -n=dic_1011 -o=/home/ubiqos/work/dataset/test2/dic_1011/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_1008 -n=dic_1008 -o=/home/ubiqos/work/dataset/test2/dic_1008/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_1009 -n=dic_1009 -o=/home/ubiqos/work/dataset/test2/dic_1009/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/dic_1004 -n=dic_1004 -o=/home/ubiqos/work/dataset/test2/dic_1004/anno.json -i /home/ubiqos/work/dataset/test2/images

python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/HylurgusLigniperda -n=HylurgusLigniperda -o=/home/ubiqos/work/dataset/test2/HylurgusLigniperda/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/DendroctonusPseudotsugae -n=DendroctonusPseudotsugae -o=/home/ubiqos/work/dataset/test2/DendroctonusPseudotsugae/anno.json -i /home/ubiqos/work/dataset/test2/images
python ./coco_tools/voc2coco.py -d=/home/ubiqos/work/dataset/test2/DebusEmarginatus -n=DebusEmarginatus -o=/home/ubiqos/work/dataset/test2/DebusEmarginatus/anno.json -i /home/ubiqos/work/dataset/test2/images
```
## 2. merge coco files

```sh
python ./coco_tools/coco_merge.py -c ./settings/cmd.yaml
```
## 3. split coco files

```sh
python ./coco_tools/coco_spliter.py --img-path=./dataset/images --json-path=./dataset/all.json --output-path=./dataset  --train-ratio=0.8
python ./coco_tools/coco_spliter.py  --json-path=/home/ubiqos/work/dataset/test2/all.json --output-path=/home/ubiqos/work/dataset/test2/ --img-path=/home/ubiqos/work/dataset/test2/images  --train-ratio=0.8

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
