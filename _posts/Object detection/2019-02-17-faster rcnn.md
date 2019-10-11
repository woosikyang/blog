---
layout: post
comments: true
title: RCNN 부터 Mask R-CNN까지 (2) Faster RCNN
categories: Object Detection
tags:
- Object Detection
---


Faster R-CNN
=======

안녕하세요 본 포스트에서는 Faster R-CNN을 다루도록 하겠습니다. Faster R-CNN은 전 버전이라 할 수 있는 Fast RCNN의 한계를 보완하고 실제 Detection시에도 빠른 속도를 보여주었기에 굉장히 주목을 받은 모델입니다. Faster RCNN이 이처럼 큰 주목을 받게 된 것은 Fast RCNN에서 오랜 속도를 만들게 한 요인인 region proposal 방식을 딥러닝 구조 안에 녹여 냈기 때문입니다. 차후 상술하겠지만 region proposal을 CPU가 아닌 GPU를 활용하여 신경망 구조 안에서 해결했기에 실제 detection에서도 0.198초라는 굉장한 속도 개선을 이루어냅니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/9odBZIP.png"></p>
<p align="center"> Faster R-CNN구조 </p>



들어가며 
---

faster r cnn은 두 개의 모듈로 구성됩니다. 첫번째는 region proposal을 하는 deep conv network 이고 두 번째는 제안된 영역을 사용하는 fast rcnn입니다. 마치 어텐션 매커니즘처럼 RPN모듈은 Fast R-CNN이 어디를 봐야 할지를 알려줍니다. 


**RPN**

Fast R-CNN의 가장 핵심적인 구조가 RPN입니다. RPN은 Conv로부터 얻은 feature map의 어떠한 사이즈의 이미즈를 입력하고 출력으로 직사각형의 object score와 object proposal을 뽑아냅니다. 이 과정은 fully conv network로 진행됩니다. 논문의 목적이 fast r-cnn의 O.D 네트워크와의 계산을 공유하는 것이기에 두 네트워크가 공통의 conv 층을 공유하는 것을 가정합니다. 실험에서는 5 층으로 구성된 ZF모델과 13개의 층으로 구성된 VGG16 모델을 사용했습니다. 

region proposal을 만들기 위해 feature map의 마지막 conv 층을 작은 네트워크가 sliding하도록 합니다. 이 작은 네트워크는 입력단으로 n*n을 받습니다. 각 슬라이딩 윈도우는 저차원으로 매핑됩니다.(ZF는 256차원, VGG는 512차원, 그후 Relu적용) 

그 후 두 개의 FCN을 통해 regression과 classification을 수행합니다. 실험에서 n=3을 적용하였고 각각 ZF모델은 171픽셀, VGG모델은 228픽셀값을 사용합니다. 

**Anchors**

각 슬라이딩 윈도우에서 다양한 후보 영역을 동시적으로 예측하는 데 이때 최대 가능 proposal을 k라 합니다. 이때의 k는 미리 정해진 파라미터입니다. 그 후 중간층을 지나 최종적으로 regression 층은 4k를, clss층은 2k만큼의 출력을 갖게 됩니다. 이처럼 후보 영역이 될 수 있는 k는 anchor라고 부르며 각 스케일과 비율에 따라 달라집니다. 실험에서는 3개의 스케일과 3개의 비율은 사용하여 k=9개의 앵커를 사용했습니다. 이처럼, 미리 정해진 앵커를 사용하는 것은 image pyramid 처럼 크기를 조정할 필요도, multi scaled slinding window처럼 filter 크기를 변경할 필요도 없는 매우 효율이 좋은 방식이 됩니다. W*H크기만큼의 conv feature map에서 WHk만큼의 앵커가 존재 합니다.





<p align="center"><img width="500" height="auto" src="https://i.imgur.com/bcij0ZI.png"></p>
<p align="center"> RPN </p>



앵커는 translation invariant (이동불변성)이라는 특징을 갖기에 레이블의 이동에도 강건한 특징을 갖습니다. 또한 파라미터 수를 감소시켜 계산을 덜 복잡하게 만들어줍니다. 

**RPN Loss Function**

RPN의 학습은 object인지 아닌지, regressor의 값 구하기로 두가지로 나뉩니다. 아래는 RPN의 로스 함수를 보여줍니다. $$ N_\cls $$ 는 미니배치에 사용된 roI를, $$ N_\reg $$는 실험에 사용한 RoI의 개수를 의미합니다. RPN에서 가장 높은 IOU, 혹은 IOU 0.7 이상을 레이블이 있는 포지티브, IOU 0.3이하를 negative라고 부르며 이를 활용하여 학습이 진행됩니다.  



<p align="center"><img width="500" height="auto" src="https://i.imgur.com/Y6ES5fA.png"></p>
<p align="center"> Loss fucnction </p>


Faster R-CNN 학습과정
---


1. RPN은 ImageNet을 사용하여 학습된 모델로부터 초기화되어 region proposal task를 위해 end to end로 학습됩니다. 
2. 윗 단계에서 학습된 RPN을 사용하여 Fast R-CNN 모델의 학습을 진행합니다. (초기화는 ImageNet의 학습 모델로)
3. 초기화를 위의 네트워크를 사용하여 RPN을 학습하는데 공통된 Conv layer는 고정하고 RPN에만 연결된 층만 학습합니다. 
4. 공유된 Conv layer를 고정시키고 Fast R-CNN의 학습을 진행합니다. 



<p align="center"><img width="500" height="auto" src="https://i.imgur.com/xYCyHKY.png"></p>
<p align="center"> Faster R-CNN 학습과정 </p>



실험 및 결과 
---

PASCAL VOC 2007, 2012 및 MSCOCO 데이터를 사용하였습니다. 
합성곱신경망 모델로 ZF와 VGG16 모델을 사용합니다. 
다양한 실험을 통해서 RPN,Conv 공유, multi task가 성능 향상을 가져온 다는 것이 확인되었습니다. 무엇보다 SS보다 훨씬 빠른 속도를 보입니다. 데이터를 많이 추가할수록 성능 향상을 이룰 수 있었다는 점도 보였습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/d4GaTrH.png"></p>
<p align="center"> 실험결과 </p>



<p align="center"><img width="500" height="auto" src="https://i.imgur.com/m3dVQRh.png"></p>
<p align="center"> 속도비교 1 </p>


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/c2eNWkc.png.png"></p>
<p align="center"> 속도비교 2 </p>


최종적으로, Faster R-CNN은 RPN이란 구조를 통해 cost-free한 region proposal 방법론을 제안하였으며, 빠른 속도와 높은 정확도의 object detection을 이루어냈습니다. 