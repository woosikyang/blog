---
layout: post
comments: true
title: Bottom-Up and Top-Down Attention for Image Captioning and VQA
categories: VQA
tags:
- VQA

---

안녕하세요 오늘은 Bottom-Up and Top-Down Attention for Image Captioning and VQA 을 리뷰하고자 합니다. VQA는 제가 그동안 쭉 관심을 가져왔지만 막상 어떠한 방식으로 이루어지는지는 몰랐던 분야입니다. 이번 논문을 통해서 최신 VQA가 어떻게 진행되는지를 알 수 있어서 좋았습니다. 앞으로도 VQA, Image captioning과 같은 이미지와 NLP의 조합을 자주 다룰 예정입니다. 그럼 논문 요약 진행하겠습니다. 


들어가며
-------

이미지와 언어를 이해하는 이미지 캡셔닝, VQA는 컴퓨터 비전과 자연어처리 모두를 아우러야 한다는 점에서 관심을 끌고 있습니다. 무엇보다 visual attention 메커니즘이 사용되었다는 특징이 있는데요, visual attention이란 이미지의 어떠한 지역을 집중해야 하는지를 찾는 메커니즘으로 보시면 되겠습니다. 

이미지를 이해하기에는 크게 두 방법이 있습니다. 첫번째는 이미지 전체를 보고 그 이미지에서 Task에 걸맞는 특징을 찾는 방법입니다. 이를 Top-down 방식이라 합니다. 두 번쨰는 이미지의 픽셀 단위부터 조금씩 파악하여 특징을 찾는 방식이며 이를 Bottom-up 이라고 합니다. 현재 대부분의 visual attention 메커니즘은 top-down 방식에 근거하였으며 이는 인간의 인지 시스템과 유사하여 좋은 결과를 가져왔습니다. 

그러나 Top-down 방식은 이미지의 어떠한 부분을 정확히 봐야 할지에 대한 근거가 부족하다는 점에서 주어진 이미지의 특징을 백퍼센트 활용하지 못한다는 한계를 갖습니다. 아래 사진을 본다면 인간의 인식처럼 task에 적합한 부분을 봐야 하는 경우가 기존의 top-down 방식으로는 한계를 갖습니다. 기존의 방식으로는 그저 동일 크기의 grid로만 판단을 하게 되기 때문입니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/3hHyTRW.png"></p>

<p align="center"> 그림 1 </p>

위의 그림과 같이 적절한 bottom-up 모델을 활용해서 주어진 이미지의 어떤 지역을 특정해서 봐야할지를 안다면 이미지에 대한 정확한 이해가 가능해질 것입니다. 따라서 본 논문은 기존의 Top-down 방식과 개선된 bottom-up 방식의 결합을 통해 더 높은 수준의 Image Captioning, VQA 모델을 만드는 데에 있습니다. 

---

**Bottom-Up Attention Model**

본 논문에서는 Top-down방식과 결합한 Bottom-up attention model로 Faster R-CNN을 제안합니다. Faster R-CNN은 RPN이라는 후보 영역 제안 신경망과 Object Detector인 Fast R-CNN의 결합을 통해 빠른 속도와 높은 성능의 Object Detection을 가능케 한 모델입니다. ImageNet으로 pre-trained된 Resnet-101 모델을 활용하여 Faster R-CNN을 진행하였고 이를 통해 주어진 이미지에서 detect할 지역과 label을 설정하였습니다. 다만 기존의 Faster R-CNN에서 attribute를 예측하는 attribute predictor를 추가함으로써 후보 영역의 class를 더 잘 예측하도록 하였다는 차이점이 있습니다.  


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/g5XHsYj.png"></p>

<p align="center"> Faster R-CNN 예시 </p>



---

논문에서는 주어진 bottom-up attention 모델을 각자 서로 다른 Top-down 모델과의 결합을 통해 Captioning과 VQA Task를 진행합니다. 


**Captioning Model**

캡셔닝에서는 기존 출력된 부분 시퀀스를 문맥으로 사용하며, 각 caption generation에 feature 가중치를 계산하기 위해서 soft top-down attention을 사용합니다. 캡셔닝 모델에서는 bottom-up attention이 없어도 좋은 성능을 보인다고 합니다.  두개의 LSTM 층으로 구성되어 있으며 각 층이 서로 다른 부분을 담당합니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/3SMD9Vq.png"></p>

<p align="center">Caption 모델의 구조 </p>


위 그림의 Caption 모델의 구조를 보여줍니다. 첫번째 층은 Top-Down attnetion LSTM 층으로 인풋으로는 이전 시점의 language LSTM / 이미지의 mean pooling 값 / 이전까지 생성된 단어, 이렇게 세 가지를 사용합니다. 이 세 가지 인풋이 소프트 어텐션을 통해 $$ h_t $$ 를 생성하며 다시 한번 bottom-up으로 구한 이미지의 mean pooling값을 활용하여 두번째 LSTM 층인 Language LSTM으로 들어갈 인풋 $$ v_t^{\^} $$ 를 생성하게 됩니다. 

두번째 LSTM층을 지난 최종 출력 y는 일련의 단어가 되며 각 시점의 조건부 분포의 곱을 통해 최종 출력 문장이 결정됩니다. 논문에서는 다양한 조건에 맞춘 loss function을 제공합니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/9jIy0Tm.png"></p>

<p align="center"> 최종 문장 출력 공식 </p>

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/RWFKIfm.png"></p>

<p align="center"> 다양한 loss 함수 </p>

**VQA Model**

VQA 모델에서는 질문인 Question representation을 문맥으로 사용하고 마찬가지로 soft attention을 사용합니다. 전체 모델 구조는 아래 그림과 같습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/pgOA7bM.png"></p>

<p align="center"> VQA 모델 구조 </p>

그림과 같이 이미지와 질문을 모두 사용하는 joint multi-modal embedding 구조입니다. 이미지 feature를 생성할때 Question의 representation을 활용하며 최종적으로 question과 image feature의 concat를 통해서 후보 답변에 해당하는 예측 점수를 계산합니다. 

---

**실험결과**

MSCOCO 와 VQA v2.0 데이터셋을 활용하여 실험을 진행하였고 예상대로 기존의 방법론보다 더 앞선 성능을 보여줍니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/IOBPP8X.png"></p>

<p align="center">캡셔닝 모델 결과 </p>

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/6dUcGag.png"></p>

<p align="center">VQA 모델 결과 </p>

정량적 분석 측면에서도 이미지에서 어떤 부분을 봐야할지에 대한 파악이 잘 이루어졌음이 확인되었습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/33T4jcI.png"></p>

<p align="center">생성된 캡셔닝의 어텐션 지역</p>

---

**결론**

본 논문은 기존의 Top-down 모델에 더해 bottom-up 모델을 결합한 방식을 제안하였습니다. 이를 통해서 attention이 더 자연스럽게 task에 반영될 수 있도록 구현하였고 이미지 캡셔닝과 VQA 모델에서 좋은 성능을 보여줍니다. 



{% if page.comments %} <div id="post-disqus" class="container"> {% include disqus.html %} </div> {% endif %}
