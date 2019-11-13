---
layout: post
comments: true
title: From Recognition to Cognition : Visual Commonsense Reasoning
categories: VQA
tags:
- VQA

---

안녕하세요 오늘 리뷰할 논문은 From Recognition to Cognition : Visual Commonsense Reasoning 입니다. 기존의 VQA 모델이 단순 질문과 정답을 찾는 것에 그쳤다면 이 논문은 새로운 데이터셋을 제공하면서 질문에 대한 정답 찾기와 그 정답을 찾는 이유까지를 학습하도록 설계되어 있습니다. 



VCR
---

논문에서는 사람들의 상식을 QA에 적용하고자 합니다. 주어진 질문에 대하여 보통 사람들의 정답은 거의 일치하게 되는데, 이는 사람들이 갖고 있는 상식에 근거하기 때문입니다. VCR 데이터셋은 이러한 상식, 즉 정답으로 유도하게 되는 근거를 학습하고자 rationale이라는 '이유' 데이터셋을 추가하였습니다. 이를 통해서 기존의 VQA dataset이 갖고 있는 학습의 한계를 보완했습니다. 



데이터셋 구성
-------

1. 29만개의 객관식 문제 

2. 29만개의 객관식 문제에 대한 이유

3. 11만개의 이미지 

4. 어렵고 다양한 문제 및 비편향되고 다양한 선택지로 구성

5. 정답 문장의 평균 단어 개수 : 7.5 / 이유의 평균 단어 개수 : 16


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/eS1mqh6.png"></p>

<p align="center"> 데이터셋 구성 </p>



적용한 모델(R2C)
---

본 논문에서 제안하는 모델의 이름은 R2C, Recognition to Cognition Networks입니다. 이미지를 보고 주어진 질문에 답하기 위해서는 여러 추론 과정이 진행됩니다. 첫번째로 질문과 정답을 확인해야 하며 그 과정에서 주어진 이미지와 연계하여 사고해야 합니다. **(Grounding)** 두번째로는 질문으로부터 얻은 정보와 이미지로부터 얻은 정보, 정답에 대해 통합적인 판단이 이루어집니다. **(Contextualization)** 마지막으로 통합적인 판단에 대한 상식선에서의 근거가 가치 수행되게 됩니다. **(Reasoning)** 모델 또한 위와 같은 추론 과정을 통해 학습되도록 구축되었습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/NP8hcFJ.png"></p>

<p align="center"> R2C 구조 </p>

1. Grounding

Grounding을 위해 구축된 모델에서는 각 sequence의 token에 대해 이미지와 질문, 정답에 대한 joint representation을 학습하게 됩니다. 논문에서는 grounding module에서 Bidirectional LSTM을 적용하였고, CNN을 사용하여 이미지의 객체 정보를 추출했는데 bounding box로부터 얻은 ROI 값을 활용하였습니다. 이미지 객체의 레이블 정보까지 사용했습니다. 


2. Contextualization 

질문과 정답에 대한 representation을 얻으면 attention 메커니즘을 통해 정답과 질문, 정답과 미지간의 context 정보를 추출하게 됩니다. 

3. Reasoning

마지막으로 bidirectional LSTM의 인풋으로 위해서 구한 attened 질문, attened 이미지 및 정답을 사용하여 multi-class cross entropy loss로 문제를 해결합니다. 질문과 정답의 임베딩 방법론으로는 BERT를, 이미지 정보 추출에는 Resnet50을 사용했습니다. 


결과
---

실험은 크게 Q->A (질문과 정답), QA->R (질문, 정답과 이유), Q->AR(질문, 정답과 이유) 세 가지로 진행되었습니다. 아래는 실험 결과입니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/bSNHwcv.png"></p>

<p align="center"> Ablations of model  </p>



기존의 대표적인 방법론인 BOTTOMUPTopDown에 비해 확실히 높은 성능을 보여줍니다. 이미지를 사용하지 않고 BERT만 사용하여 질문과 정답을 학습할 경우에도 생각보다 높은 성능을 갖는다는 점도 눈여겨 볼만 합니다. BERT방법론 자체가 강력한 모델임을 보여준다고 볼 수 있겠습니다. 


최종 결론 
---

VCR은 기존의 VQA Dataset이 갖고 있는 한계를 보완했다는 점에서 의미가 있으며 인간의 사고와 유사한 방식으로의 학습을 유도한다는 점에서 진정한 VQA task 해결에 가까워질 수 있는 데이터셋입니다. 무엇보다도 이번 2019 ICCV에서 논문에 대한 발표가 있었는데 개인적으로 공부하던 논문이 실제 학회에서 발표되는 모습을 보면서 저 또한 연구자로 의미 있는 발표를 진행하고 싶은 마음이 활활 타오르는 계기였습니다. 앞으로 이 데이터를 활용한 연구를 진행하여 유의미한 향상을 이루어내고 싶습니다. 

