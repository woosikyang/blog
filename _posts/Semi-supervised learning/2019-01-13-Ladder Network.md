---
layout: post
comments: true
title: Semi-supervised learning with Ladder Network 
categories: Semi-supervised learning
tags:
- Semi-supervised learning
---

안녕하세요 오늘 리뷰할 논문은 준지도학습 분야의 Semi-supervised learning with Ladder Network 입니다. 유투브에서 고려대학교 DSBA 연구실 채널에 가시면 제가 이 논문을 가지고 발표한 자료를 시청하실 수 있습니다. 아무쪼록 이 포스트를 보시는 분에게 도움이 되기를 바랍니다. 그럼 논문 리뷰 시작하겠습니다. 포스트에서 사용한 그림의 출처는 제가 만든 자료와 리뷰할 본 논문입니다. 


들어가며
-------

본 논문은 2015년도, 즉 동년도에 같은 저자가 저술한 From neural PCA to deep unsupervised learning을 발전시킨 논문입니다. 두 논문 모두 준지도학습을 위한 Ladder Network라는 새로운 구조를 도입했으며 이전 논문은 단순히 비지도학습에서만 실험을 진행했다면 오늘 소개할 논문은 지도학습에서 실험을 진행했다는 차이점이 있습니다. 이 논문에서는 기존에 비지도학습이 지도학습을 돕기 위해 사용될때 단순히 pre training 단계에서만 사용되는 모습을 지양하고 마치 supervised learning처럼 계속 학습을 진행하여 비지도학습 부분에서도 데이터가 가지는 다양한 특징을 활용할 수 있었다는 특징을 보입니다. 

제가 파악한 Ladder Network의 도입을 위해 진행된 생각의 흐름은 다음과 같습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/nLOQIGm.png"></p>

<p align="center"> 그림 1 </p>

일반적으로 준지도학습을 위해서는 잠재 변수 모델이 선호됩니다. 잠재 변수 모델은 주어진 데이터의 특징을 표현할 수 있는 핵심적인 변수가 존재한다고 가정합니다. 지도학습과 비지도학습의 차이는 학습에 필요한 label의 여부만 나기에 잠재 변수 모델을 사용한다면 지도학습과 비지도학습을 효과적으로 사용할 수 있게됩니다. 

그러나 단층 잠재 변수 모델은 데이터의 계층적인 특징을 효과적으로 반영할 수 없습니다. 이는 사람의 얼굴을 구분할때 계층적 모델은 얼굴의 윤곽부터 구체적인 눈, 코, 입과 같은 지엽적인 정보부터 사람의 얼굴을 구성하는 핵심적인(불변하는) 특징까지 단층적인 모델로 표현할 수 없다는 것과 동일합니다. 따라서 여러 층을 활용하는 계층적 잠재 변수 모델(Hierarchical latent variable model)이 이상적입니다. 

다만 계층적 잠재 변수 모델은 계산하는 데 있어서 많은 단점을 가지고 있습니다. 따라서 본 논문에서는 비지도학습 방법론인 Autoencoder의 구조를 사용하여 데이터의 계층적인 특징을 활용하고자 하였습니다. 기본적인 오토인코더는 단층 구조이기에 단층 잠재 변수 모델과 같은 단점을 가지고 있으므로 여러 개의 layer를 쌓아서(Stacked) 활용합니다. 

Stacked Autoencoder는 그 구조면에서 본다면 앞서 언급한 Hierarchical latent variable model과 유사합니다. 그러나 큰 단점이 있는데요. 오토인코더에서 아래층에서 위의 층으로의 연결은 확률적(Stochastic)이지 않고 결정적(Deterministic)하다는 점에 있습니다. 오토 인코더에서 층 간의 연결이 매핑함수로 구성되어있기 때문입니다. 

본 논문에서는 이러한 Autoencoder의 단점을 개선하고 데이터의 계층적인 특징을 활용하기 위해서 인코더와 디코더를 수평적으로 연결하여 네트워크를 구성합니다. 마치 그 모양이 사다리를 닮아서 Ladder network라고 부르게 되는 것이지요. 아래 그림 2는 지금까지 언급한 세 구조를 단편적으로 보여주고 있습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/GCKMagz.png"></p>

<p align="center"> 그림 2 </p>


결론적으로 말하자면, 효율적인 준지도학습을 위해서 Hierarchical latent variable 모델의 특징을 반영하는 Autoencoder Network를 구축하였고 그 과정에서 더욱 효과적인 학습을 위해 Denoising 기법을 적극 활용한 결과물이 바로 Ladder Network가 되겠습니다. 



Denoising
-------

본 논문에서 또 다른 핵심이 되는 부분이 바로 Denoising입니다. Denoising, 즉 잡음 제거 방식은 잡음을 추가한 데이터를 학습하여 데이터가 가지고 있는 본래의 고유한 특징을 더 잘 찾기 위한 방법입니다. 크게 Denoising frame work 와 Denoising Autoencoder 부분으로 얘기할 수 있습니다. 만일 주어진 데이터 X에 대한 확률 모델을 알고 있다면, 효과적인 샘플링 및 Denoising function을 활용할 수 있을 것입니다. 즉, optimal한 denoising function을 사용하는 것이 좋은 샘플링 및 데이터 x를 파악하는데 도움이 될 수 있다는 의미가 되는데 확률 모델을 구하는 것보다 Denoising 기법을 사용하는 것이 더 쉽기 때문에 Ladder Network에서 사용했다고 이해하면 되겠습니다. 

**Denoising Autoencoder**

Denoising Autoencoder는 기본적인 Autoencoder 구조에서 Input data에 노이즈를 추가하여 학습을 진행하는 비지도학습을 말합니다. 오토인코더가 인풋을 잘 복원하는 아웃풋을 학습한다면 dAE는 노이즈가 추가된 인풋 데이터가 압축(encode)되고 복원(reconstruct)되는 과정 후 생성된 output이 노이즈가 없는 최초 상태의 인풋과 최대한 유사하도록 학습하게됩니다. 흔히 알려진 예로 안개 속에서 사람을 구별하는 예시를 들 수 있습니다. 안개 속에서 사람이 우산을 쓰고 있더라도 우리는 그 사람을 판별할 수 있습니다. 안개로 인해 우리의 시각이 조금 방해받지만, 사람임을 판별할 수 있는 핵심 요소는 반대로 강하게 작용하기 때문이죠. 이 같은 원리를 오토인코더에 적용한 방법론이 dAE인 것입니다. 
Ladder network에서 dAE구조는 마치 지도학습과 같이 학습이 진행됩니다. 즉, 노이즈를 추가한 오염된 경로(corrupted path)의 아웃풋과 타겟값의 loss를 줄이는 방향으로 학습이 진행됩니다. 

**Denoising Source Separation framework**

Denoising Source Separation framework (DSS framework)는 저자의 관련 연구인 Denoising Source Separation에서 사용된 Denoising 도입 방식을 의미합니다. dAE가 output,즉 출력물과의 관계를 학습한다면 DSS framework를 통해서 잠재변수간의 관계를 학습합니다. Denoising의 특징에 맞게 노이즈가 추가된 데이터로부터 만들어지는 잠재 변수는 실제 깨끗한(clean path)로 인해 만들어지는 잠재 변수와 최대한 유사하게 학습됩니다. DSS framework에서는 잠재변수(z)의 normalization을 요구하는데 논문에서는 이를 Batch Normalization으로 해결합니다. 


Ladder Network
-------

다음은 Ladder Network의 구조를 나타냅니다. 앞서 언급했듯이 corrupted path, clean path, denoising path로 구성되어 있으며 학습 또한 지도학습과 비지도학습이 결합되어 진행됩니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/qn1z3Zw.png"></p>

<p align="center"> 그림 3 </p>


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/OqucsZE.png"></p>

<p align="center"> 그림 4 </p>



Implementation of the model
-------

Ladder Network를 모델에 도입하는 과정은 세 단계로 이루어집니다. 

- 1: Encoder로 지도 학습을 하는 feedforward 모델 구축

feedforward 모델에서는 지도학습과 비지도학습이 병렬적으로 진행됩니다. 앞서 DSS framework를 위해 batch normalization을 사용한다고 언급했습니다. 논문에서는 Batch normalization을 통해 BN이 주는 효과인 covariance shift를 줄이는 동시에 일반화 가정을 만족시킬 수 있다고 언급합니다. Feedforward 모델 구축을 통해서 지도학습과 비지도학습이 병렬적으로 진행되며 지도학습 측면에서는 corrupted path에서 생성된 output이 실제 target과 유사하도록 학습이 진행됩니다. 그림 4의 파란색으로 표시된 지도학습에 해당하는 부분입니다. 

- 2: 각 층과 mapping하고 비지도학습을 돕는 decoder 구축

두 번째 단계에서는 비지도학습이 진행됩니다. 즉 1번 단계에서 encoder단에서 학습된 비지도학습 가중치가 decoder단으로 내려오면서 아래층의 잠재변수 학습에 영향을 주고 동시에 수평적으로 연결된 corrupted path의 같은 층의 정보가 영향을 주면서 clean path의 z와 유사하도로 학습이 진행됩니다. 

- 3: 모든 손실합수의 합을 최소화하는 Ladder Network 학습 

지도학습의 loss function과 비지도학습의 loss function을 합친 최종 loss function이 작아지도록 학습이 진행됩니다. 


**CNN 구조로의 확장**

지금까지의 Ladder Network의 도입은 여러 층을 가진 MLP(Multi-layer perceptron)을 사용했습니다. 그러나 Ladder Network는 CNN과 같이 기존에 존재하는 뉴럴 네트워크에 쉽게 도입할 수 있습니다. CNN 구조에서도 마찬가지로 encoder 파라미터의 흐름을 역으로 반영한 decoder를 구축하는 방식으로 쉽게 사용할 수 있습니다. 


**감마 모델**

Decoder의 가장 높은 layer만을 사용하는 모델을 감마 모델이라고 부르며 상대적으로 더 간단한 모형이지만 실험에서도 좋은 성능을 보이고 있습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/pBxgvc6.png"></p>

<p align="center"> 감마 모델 </p>


실험 결과 
-------

실험에서는 MNIST 데이터와 CIFAR-10 데이터를 사용하였고, MLP 네트워크와 서로 다른 CNN 네트워크를 사용하였습니다. CIFAR-10 데이터에서는 감마 모델만을 사용한 실험을 진행했습니다. 실험에 따라 사용한 label의 수에 변화를 주어 지도학습을 진행했지만 비지도학습은 전체 데이터를 모두 사용하였습니다. 공통적으로 기존의 결과와 비교했을때 SOTA의 성능을 보임을 확인할 수 있습니다. MNIST의 경우에는 label을 100개만 사용했을 경우에도 좋은 실험 결과를 보여주고 있음을 알 수 있습니다. MNIST데이터에 대해 워낙 좋은 성능을 보이는 뉴럴 네트워크 구조들이 많지만 label수를 적게 사용해도 이 정도의 성능을 보인다는 점은 주목할만 하겠습니다. 또한 감마 모델만을 사용한 CIFAR-10에서는 감마 모델만으로도 충분히 좋은 성능을 보이고 있음이 확인됩니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/8wBhqx9.png"></p>

<p align="center"> MNIST + MLP </p>


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/yT52obf.png"></p>

<p align="center"> MNIST + CNN </p>


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/FWrnwS7.png"></p>

<p align="center"> CIFAR-10 + CNN </p>


마치며 
-------

아직까지 학습에 필요한 데이터를 쉽게 구하기는 쉽지 않습니다. 작년 하반기, 몇달전만 하더라도 저희 연구실에서 양질의 데이터를 구축하기 위해 연구원들 모두가 개인시간을 꼬박 할애하며 끝없는 작업을 진행했던 기억이 생생합니다. 앞으로도 제대로된 label이 달린 데이터를 구하기란 쉽지 않을 것입니다. 이런 측면에서 비지도학습의 풍부한 정보를 지도학습의 Task의 걸맞게 적용할 수 있다면 딥러닝은 한층 더 진보할 수 있을 거라 믿습니다. Ladder network는 연구의 목적과 더불어 목적 해결을 위해 창의적으로 수평적 연결을 시도함으로써 재미있게 읽을 수 있었던 논문이었습니다. 동시에 이 논문을 공부하면서 수학적인 베이스에 대한 필요성도 절실히 느꼈는데 이러한 부족함과 창의성을 향후 추가적인 노력을 통해 채울 수 있기를 바라고 있습니다.