# cauda_project
Aphid homology analysis AI development project with mask rcnn 


```sh
python ./coco_tools/voc2coco.py -d=/home/ubiqos-ai2/work/datasets/bitles -n=dic_1009 -o=./temp/dic_1009/anno.json

#merge json files

PYTHONPATH=./coco_tools python./coco_tools/merge.py

python ./coco_tools/coco_spliter.py --img-path=/home/ubiqos-ai2/work/datasets/bitles/images --json-path=./temp/all/anno.json --output-path=./temp/all --train-ratio=0.8 --test-ratio=0.1

# config 파일 만들기 
python make_train_cfg.py --eval-period=100 --epoch=300 --dataset-root=./temp  --dataset-name=all  --batch=6 --base-config-file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml


# train
python train_net.py --config-file ./configs/dic_1009/config.yaml --dataset-path=./temp --dataset-name=dic_1009 --image-root=/home/ubiqos-ai2/work/datasets/bitles/dic_1009/voc --num-gpus 1


```

## 탠서보드 

--logdir 옵션으로 학습결과물이 출력되는 디랙토리를 지정한다.<br>

```sh
tensorboard --logdir /home/ubiqos-ai2/work/visionApp/cauda_project/output/all
```
http://localhost:6006 으로 접속한다.<br>
