# cauda_project
Aphid homology analysis AI development project with mask rcnn 


```sh
python ./coco_tools/voc2coco.py -d=/home/ubiqos-ai2/work/datasets/bitles -n=dic_1009 -o=./temp/dic_1009/anno.json

python ./coco_tools/coco_spliter.py --img-path=/home/ubiqos-ai2/work/datasets/bitles/dic_1009/voc --json-path=./temp/dic_1009/anno.json --output-path=./temp/dic_1009 --train-ratio=0.8 --test-ratio=0.1

# config 파일 만들기 
python make_train_cfg.py --eval-period=100 --epoch=300 --dataset-root=./temp  --dataset-name=dic_1009  --batch=6 --base-config-file=COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml


```

