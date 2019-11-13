---
layout: post
comments: true
title: Tips and tricks for VQA-learnings from 2017 challenge
categories: VQA
tags:
- VQA

---

안녕하세요 오늘 리뷰할 논문은 Tips and tricks for VQA-learnings from 2017 challenge 입니다. VQA는 높은 관심에 비해 굉장히 어려운 task이므로 아직 확실한 정복이 이루어졌다고 말하기 어려운 분야입니다. 본 논문은 다양한 시도가 이루어진 과거의 논문들을 통해서 강건한 성능의 VQA모델을 제시하고, 어떠한 특징이 VQA에 효과적인지를 말하고 있습니다. 

주요 특징
-------

1. 기존의 single-label 소프트맥스가 아닌 sigmoid output을 사용하여 질문당 다수의 정답을 활용하였습니다. 

2. classification이 아닌 soft score값을 사용한 regression 문제로 truth target과 학습이 진행되도록 구성했습니다. 

3. 모든 비선형 층에 gated tanh activation을 사용했습니다. 

4. 기존의 CNN 방식이 아닌 bottom-up attention으로부터 얻은 image feature을 사용했습니다. 

5. 출력층의 가중치 학습 초기화를 위해 후보 정답들로부터 pretrained된 representation값을 사용하였습니다.  

6. 큰 미니배치와 smart shuffling을 사용한 SGD를 구성했습니다.  


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/xzowGEQ.png"></p>

<p align="center"> 제안하는 모델 성능 </p>


기존 VQA Approch 
---

논문에서는 기존 VQA 접근 방식을 소개합니다. 바로 1. question answering을 classification 문제로 바라보며 2. 이 문제를 joint embedding을 활용한 딥 뉴럴 네트워크로 풀고 3. end to end로 지도학습을 통해 학습을 진행하는 방식이지요. 흥미롭게도 좋은 하이퍼파라미터를 사용한다면 굉장히 간단한 모델도 좋은 성능을 보여줄 수 있음이 확인되었습니다. 본 논문에서도 위에서 말한 주요 특징만을 사용해서 간단한 모델에서도 좋은 성능을 가져왔다는 점을 강조합니다. 


제안하는 모델
---

본 논문에서 제안하는 모델은 아래 그림과 같습니다. 어디서 많이 보신 그림 같으신가요?? 저번 포스팅에서 소개한 bottom-up & top-down attnetion for image captioning and VQA 와 굉장히 유사하다는 것을 알 수 있습니다. 그림에 나와 있듯이 맨 처음에 설명한 조금의 variation을 통해서 좋은 성능을 획득했다는 점을 확인할 수 있습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/qrULeDo.png"></p>

<p align="center"> 제안하는 모델 </p>


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/qrULeDo.png"></p>

<p align="center"> Bottom-Up Attention Model </p>


Ablations of model 
---

본 논문에서 눈길을 끈 점은 바로 network ablation table이었습니다. VQA task에 있어서 각자의 architecture가 얼마나, 어떠한 영향을 가져올 수 있는지를 생각해 볼 기회를 준다는 점에서 굉장히 좋은 자료라 생각합니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/wqh7rAI.png"></p>

<p align="center"> Ablations of model  </p>

최종 결론 
---

본 논문은 좋은 성능을 보이는 VQA 모델을 제시합니다. 다만 전혀 새로운 구조의 모델이 아닌, 기존에 제안된 간단한 모델을 사용하며 이를 통해서 복잡한 모델이 아닌 VQA에서 design choice와 hyperparameter choice, detailed implementation 만으로도 강건한 성능을 확보할 수 있다는 것을 보여줍니다. 본 논문에서 직접 밝혔듯이 제공되는 결과들은 충분히 좋은 baseline으로 사용할 수 있으며 앞으로 VQA에 대한 깊은 이해와 새로운 시도를 통해 더 나은 모델을 구축할 수 있도록 해야겠습니다. 지금까지 읽어주셔서 감사합니다. 
