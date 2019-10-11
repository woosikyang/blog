---
layout: post
comments: true
title: learning by association - a versatile semi-supervised training method for neural networks
categories: Semi-supervised learning
tags:
- Semi-supervised learning
---


안녕하세요 오늘 다룰 논문은 learning by association - a versatile semi-supervised training method for neural networks 입니다. 2017 CVPR에서 발표된 논문이며 
본 블로그에 올라올 Ladder network와의 비교가 진행된 논문입니다. 


들어가면서 
-------

본 논문은 인간이 학습하는 방식을 참고한 신경망 구조를 제안합니다. 인간은 데이터간의 연관성(Association)를 통해서 학습이 가능합니다. 어린 아이의 경우에도 몇 개의 강아지 사진을 보고 강아지란걸 학습한다면 다른 강아지 사진을 보더라도 그것이 강아지인지 아닌지를 알 수 있을 것입니다. 반면 인공신경망의 경우, 좋은 성능을 보이는 구조를 학습하기 위해서는 학습에 사용되는 모든 데이터마다 그에 해당하는 레이블이 존재해야 합니다. 본 논문은 인간이 학습하는 연관성의 특징을 네트워크에 적용시킨 결과물입니다(Learning by association). 

Learning by association의 간략한 과정은 다음과 같습니다. 하나의 배치에 해당하는 label 데이터와 unlabel 데이터의 임베딩을 만듭니다. 그 후 label 배치의 샘플로부터 Imaginary walker가 unlabel 배치의 샘플로 전달됩니다. 이 전이과정(transition)은 각각의 임베딩의 유사도(similarity)에서 얻은 확률 분포에 따라 진행되며 이것을 association이라 부릅니다. 

이러한 association이 합당하게 일어나는지 평가하기 위해서 임베딩간의 유사도에 따라 label 배치로의 역전환 과정이 진행됩니다. 만일 처음의 class와 동일하게 판별한다면 두 배치가 유사한 class임을 알 수 있을 것입니다. 네트워크의 목적은 서로 연관성이 없는, 즉 다른 class의 데이터들간의 특징(essence)를 잘 잡아내는 것에 있습니다. 

Learning by Association 
-------

본 논문은 같은 class에 속한다면 좋게 임베딩된 결과물간의 높은 유사성이 있을 것임을 가정합니다. 두 데이터가 CNN을 통해서 임베딩된 벡터로 출력되며 A->B, B->A로 왔다가는 walker를 통해 두 데이터간의 관계를 파악합니다. 자세한 과정은 아래 Figure1과 같습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/QcgW47n.png"></p>

<p align="center"> Figure 1 </p>


따라서 learning by association의 목적은 A->B->A로 되는 과정에서 같은 클래스의 두 데이터의 확률을 최대화 하는것입니다. 

우선 두 A,B 임베딩간의 유사도를 계산하는 식을 아래와 같이 정의합니다. 본 논문에서는 가장 좋은 결과를 보인 내적 기법을 사용했지만 다른 방법론도 가능합니다.

$$
M_{ij} := A_i \dot B_j
$$

그 다음으로 계산한 유사도를 A에서 B로의 전이 확률(transition probabilites)로 변환합니다. 

$$
P^{ab}_{ij} = P(B_j|A_i) := (softmax_{cols}(M))_{ij} = exp(M_{ij}) / \sum_{j'}exp(M_{ij'})
$$

반대 방향의 전이 확률(P^{ba}) 또한 M을 M^T로 대체하여 구하며 두 경로의 왕복 확률(round trip probability) 를 구할 수 있습니다. 

$$
P^{aba}_{ij} := (P^{ab}P^{ba})_{ij} = \sum_k P^{ab}_{ik}P^{ba}{kj}
$$

최종적으로 correct walk에 대한 확률은 아래와 같습니다. 

$$
P(correct walk) = \frac{1}{|A|} \sum_{i \sim j} P^{aba}_{ij}
$$

CNN구조를 따로 진행한 후 walker를 추가하는 구조이기에 여러 Loss를 사용합니다. 

$$
L_{total} = L_{walker} + L_{visit} + L_{classification}
$$

각 Loss들에 대한 설명은 다음과 같습니다. 

+ Walker loss

연관성을 알아보는데 있어서 중요한 것은 같은 class를 유지해야한다는 점입니다. 같은 class여야 그 속에서 찾는 관계가 의미가 있기 때문입니다. 따라서 walker loss에서는 부정확한 walk에 대해서 패널티를 부가하며 같은 class에 대해 uniform distribution이 되도록 독려합니다. 자세한 공식은 아래와 같으며 H는 cross entropy를 의미합니다. 

$$
L_{walker} := H(T,P^{aba}) 
$$

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/Lx8WRSQ.png"></p>


+ Visit loss

unlabel 샘플들을 효과적으로 사용하기 위해서는 쉽게 판별할 수 있는 샘플이 아닌 모든 샘플을 사용(visit)해야하며 이를 통해 더 일반화된 임베딩을 얻을 수 있습니다. 

$$
L_{visit} := H(V,P^{visit}) 
$$

$$
P_j^{visit} := <P_{ij}^{ab}>_i
$$
$$
V_j = := 1/|B|
$$

+ Classification loss

임베딩 결과물을 class로 판별하는 기존에 사용하는 네트워크에서의 loss를 classification loss라 부릅니다. 

Experiments
-------

**MNIST Dataset**


MNIST 데이터를 사용하여 세세한 finetuning을 거치지 않은 간단한 모델의 실험 결과는 아래 Table 1과 같습니다. vanilla한 구조만으로도 좋은 성능을 보입니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/Iqkh2iD.png"></p>

<p align="center"> Table 1 </p>


아래 Figure 2는 실험이 진행되면서 연관성을 탐색하는 과정의 발전을 보여줍니다. 
학습 초기에는 임베딩이 좋지 않아 다른 class에도 왕복하는 모습을 보이지만 학습이 완료된 후에는 동일한 class별로 왕복이 이루어져 좋은 학습이 진행되었음을 확인할 수 있습니다.  

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/erBYej9.png"></p>

<p align="center"> Figure 2 </p>


Figure 3은 MNIST 데이터에서 실험 결과 Test error가 어디에서 나타났는지를 보여줍니다. 왼쪽 아래와 같이 틀렸을 경우에도 사람이 판별하기에도 애매한 label이 보임을 확인할 수 있습니다. 즉, MNIST 데이터를 실험한 결과 사람도 속을만큼 애매한 label을 제외하고는 높은 성능을 보여준다는 것을 의미합니다.  

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/xRYlo4g.png"></p>

<p align="center"> Figure 3 </p>




**STL-10 Dataset**



Figure 4는 SLT-10 데이터를 활용한 실험 결과를 보여줍니다. 실험을 통해서 각 class에서 가장 높은 스코어의 결과물 5장을 확인할 수 있습니다. 흥미로운점은 실제 class가 존재하는 자동차와 선박의 경우에는 자동차와 선박을 정확하게 판별했을 뿐만 아니라 train data에 class가 없는 경우에도 주어진 input과 유사한(연관성이 있는) 결과물을 output으로 출력했다는 점입니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/AqF0ktP.png"></p>

<p align="center"> Figure 4 </p>


**SVHN**

아래 Table 2,3은 SVHN 데이터를 사용한 실험 결과입니다. 적은 labeled 샘플 수에도 높은 성능을 보여주며 label 데이터, unlabel 데이터를 모두 더 많이 사용할수록 좋은 학습이 이루어짐을 보여줍니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/XvAxkRe.png"></p>

<p align="center"> Table 2 </p>



<p align="center"><img width="500" height="auto" src="https://i.imgur.com/Qspo5zr.png"></p>

<p align="center"> Table 3 </p>


Table 4는 본 논문에서 제시하는 Visit loss의 효과를 보여주고 있습니다. Visit loss는 데이터 셋에 따라서 결정되어야 하며 너무 큰 visit loss의 경우 과적합의 위험이 있음을 말하고 있습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/UyFaX8w.png"></p>

<p align="center"> Table 4 </p>


**Domain Adaptation**

마지막으로 논문에서는 Donmai Adaptation 측면에서 learning by association 개념을 사용합니다. SVHN 데이터를 source로, MNIST 데이터를 target으로 했을 때, 각 데이터를 독립적으로 학습 한 경우보다 Domain Adaptaion방법론을 적용한 결과물이 가장 좋다는 것을 보여줍니다.

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/ZWPYFos.png"></p>

<p align="center"> Table 5 </p>

마치며 
-------

Deep learning에는 다양한 분야가 존재하며 최근 들어 모든 분야에서의 딥러닝에 대한 연구와 적용을 통해 발전이 이루어지고 있습니다. 다만 연구에 필요한 좋은 데이터에 비해 실제 활용되고 있는 데이터의 질과 양은 아직 많이 부족하다고 생각됩니다. 또한 이러한 상황에서 좋은 품질의 label된 데이터를 구하는 것은 더더욱 어렵습니다. 이러한 측면에서, 비지도학습과 지도학습을 결합하고자 하는 준지도학습은 저 자신에게 굉장히 흥미롭게 다가왔으며 따라서 데이터간의 연관성을 활용하여 문제를 해결하고자 하는 본 논문은 앞으로의 연구에 있어서 많은 참고가 되었습니다. 