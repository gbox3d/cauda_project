# 평가 코드 모음

## 단순 예측 테스트 코드 샘플 

사용법  
```sh
python predict.py -w ../output/all/ -i ../../../datasets/bitles/images/1653206225898.jpeg -o ../temp
```
-w  : weights path  
-i  : input image path  
-o  : output result path  


## 평가 방법

true positive  : tp , 있는것을 있다고 예측한 것  
false positive : fp , 있는것을 없다고 예측한 것  
true negative  : tn , 없는것을 없다고 예측한 것 (여기서는 사용하지않음)  
false negative : fn , 없는것을 있다고 예측한 것  




