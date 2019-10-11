---
layout: post
comments: true
title: 비선형 차원 축소(Nonliear Mapping)
categories: Dimensionality Reduction
tags:
- Dimensionality Reduction
---

Unsupervised Methods: Nonlinear Mapping
=======



안녕하세요. 고려대학교 DSBA연구실 석사과정 양우식입니다.


이번 포스트에서는 비지도학습에서 비선형 매핑 방법론인 Isomap, LLE(Local Linear Embedding), t-SNE 알고리즘을 소개하겠습니다.   

본 포스트는 고려대학교 강필성 교수님의 Business-Analytics 강의와 Nagoya 대학교 Hayashi 교수님의 강의자료, 이 외 다양한 강의를 참고하여 작성되었습니다. 또한 본 포스트에서 소개되는 코드는 고려대학교 DSBA연구실 박사과정 서승완님, 해외의 Liam Schoneveld님의 코드를 참고하였습니다.



시작하기에 앞서 비지도학습과 차원 축소에 대해서 간략히 설명하고 넘어가도록 하겠습니다. 
비지도 학습은 머신러닝 방법의 일종으로 지도학습, 강화학습과 비교했을 때 인풋, 즉 입력값에 대한 아웃풋인 정답이 존재하지 않는다는 특징을 갖습니다. 대표적으로 차원축소, 분류에 사용되며 차원 축소는 다시 선형 차원 축소, 비선형 차원 축소로 나누어집니다. 

그렇다면 차원 축소는 근본적으로 왜 필요할까요? 
우리가 다루어야 할 데이터가 변수가 매우 많은 고차원일 경우 중요하지 않은 변수로 인해 좋은 성능을 얻지 못할 수 있거나 처리해야 할 데이터의 양 자체가 굉장히 커지게 됩니다. 이때 고차원 데이터를 저차원으로 축소시켜 데이터가 가진 대표적인 특징만을 잘 추출할 수 있다면 더 좋은 성능과 효율적인 작업이 가능해집니다. 차원 축소의 대표적인 알고리즘으로는 분산을 최대로 보존하며 저차원으로 임베딩하는 주성분 분석(PCA)이 있습니다. 
그러나 주어진 데이터의 형태가 그림 1과 같이 매니폴드한 형태를 갖고 있는다면 선형 차원 축소인 주성분 분석으로는 각 색깔로 나타나는 데이터 레이블들만의 고유한 특징을 잡아낼 수 없게 됩니다.
여기서 매니폴드(manifold)란, 고차원 공간 중에 존재하는 실질적으로 보다 저차원으로 표시 가능한 도형을 의미합니다. 대표적인 매니폴드 형태인 스위스 롤을 나타내는 그림 1에서 본다면 고차원 공간으로 보여지는 스위스 롤 도형이 사실은 색깔에 따라 저차원으로 표현이 가능함을 알 수 있습니다.
결론적으로 오늘 다룰 nonlinear mapping 방법론들은 비선형적 차원 축소를 통해 저차원으로도 데이터의 특징을 잘 확보하는데에 그 목적이 있습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/orBAuKy.png"></p>

<p align="center"> 그림1 </p>

이렇게 선형 축소 방법론이 유효하지 못할 경우 Nonlinear Mapping 방법론적 접근이 필요하며 대표적으로 오늘 다룰 Isomap, LLE, t-SNE 방법론이 있겠습니다.

---

Isomap
-------

Isomap은 다차원 스케일링(MDS) 또는 주성분 분석(PCA)의 확장이자 두 방법론을 결합한 방법론으로 볼 수 있습니다. 앞서 다루었던 PCA와 MDS의 특징을 결합하여 모든 점 사이의 측지선 거리를 유지하는 더 낮은 차원의 임베딩을 추구합니다. 여기서 측지거리란, 두 측점사이의 타원체면을 따라 이루어진 거리를 말합니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/YeDG7pl.png"></p>

<p align="center"> 그림2 </p>

그림2에 따르면 두 점은 유클리디안 거리로는 가깝지만 실제 측지거리를 구할 경우 색깔이 나타내는 의미만큼 멀리 떨어져 위치함을 알 수 있습니다. 즉, Isomap 알고리즘은 두 데이터간의 실제 특징을 반영하는 거리 정보를 사용하는 효과적인 차원 축소를 추구합니다.


Isomap 알고리즘은 세 단계로 구성됩니다. 코드와 함께 설명하며 본 알고리즘에서 사용된 데이터는 그림3과 같은 스위스 롤을 생성하여 적용하였습니다. 




**1.	인접한 이웃 그래프 구축**

첫번째 단계에서는 어떤 점들이 매니폴드 상에서 서로 가까이 위치하는지를 측정합니다. 두가지 방식이 사용되는데 첫번째는 고정된 기준값인 앱실론을 기준으로 그보다 거리가 가까운 경우의 모든 두 점을 서로 연결합니다. 두번째는 자기 자신과 가장 가까운 K개의 점을 연결하는 KNN방식으로 모든 점들을 서로 연결합니다. 첫번째 단계가 진행된 후의 모습은 그림3과 같습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/wfCi78s.png"></p>

<p align="center"> 그림3 </p>

이렇게 점끼리 연결되었을때 엣지의 가중치는 두 연결된 점 사이의 유클리디안 거리가 됩니다. 



**2.	두 점 간의 최단 경로 그래프 계산**

두 점 i와 j에 대하여 두 점이 서로 연결되어 있다면 $d_G(i,j) = d_X(i,j)$
서로 연결되어 있지 않으면 $d_G(i,j)$를 무한으로 초기화합니다. 
그 후 1부터 N개 까지의 k에 있어서 점 i와 j간의 최단 거리를 의미하는 $d_G(i,j)$를 $min(d_G(i,j),d_g(i,k)+d_G(k,i))$로 변환합니다. 이 과정에서는 대표적으로 Dijkstra 알고리즘, Floyd's 알고리즘 등이 사용됩니다.



**3.	MDS 방법론을 사용하여 d차원 임베딩 구축**

$\lambda_p $ 를 행렬 $ \tau(D_G) $ 의 p번째 eigenvalue, $v^i_p$를 p-th eigen vector의 i번째 컴포넌트라 할 때, d차원의 p번째 컴포넌트에 대한 coordinate vector $y_i$를 $\sqrt{(\lambda_p)}v_p^i$로 두어 d차원의 임베딩을 구축합니다. 여기서 MDS, 즉 다차원 척도 구성법이란 거리 데이터만 주어졌을 때 그 거리를 재현하는 것처럼 좌표계를 역산하는 방법을 의미합니다. 


Isomap의 알고리즘은 크게 출력의 차원과 이웃사이즈(K) 또는 epsilon의 하이퍼 파라미터를 갖습니다. 차원의 선택은 시각화를 위해 많은 경우 2차원 또는 3차원을 사용하며 이웃의 크기를 선택하는 것은 이에 비해 더욱 복잡합니다. 만약 이웃의 수가 너무 작으면 측지선이 데이터간의 정보를 잘 담지 못하고 k가 너무 크다면 그래프 내에서 잘못된 연결이 나타나기 때문입니다. 

Isomap의 계산복잡도는 아래와 같습니다.   


$$
 O[D \log(k) N \log(N)] + O[N^2(k + \log(N))] + O[d N^2]
$$

   
+ N: 훈련 데이터 포인트의 수 
+ D: 입력 차원수
+ k: 가장 가까운 이웃의 수
+ d: 출력 차원수



그림 4는 손글씨 2에 해당하는 MNIST데이터를 Isomap을 활용하여 2차원으로 축소한 결과입니다. 아래로 내려갈수록 숫자 2의 윗부분에 동그랗게 그려지고 오른쪽으로 갈수록 숫자 2의 아랫부분이 동그랗게 그려지는 특징을 정확히 잡아냈음을 확인할 수 있습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/FwctPtZ.png"></p>

<p align="center"> 그림4 </p>



### Isomap을 활용한 예시 

MNIST 7개의 손글씨 숫자 Dataset을 Isomap에 활용하여 차원 축소 

*  Dijkstra 알고리즘 생성 

```python
class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = {}
        self.distances = {}
    def add_node(self, value):
        self.nodes.add(value)
    def add_edge(self, from_node, to_node, distance):
        self._add_edge(from_node, to_node, distance)
        self._add_edge(to_node, from_node, distance)
    def _add_edge(self, from_node, to_node, distance):
        self.edges.setdefault(from_node, [])
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = distance
def dijkstra(graph, initial_node):
    visited_dist = {initial_node: 0}
    nodes = set(graph.nodes)
    while nodes:
        connected_node = None
        for node in nodes:
            if node in visited_dist:
                if connected_node is None:
                    connected_node = node
                elif visited_dist[node] < visited_dist[connected_node]:
                    connected_node = node
        if connected_node is None:
            break
        nodes.remove(connected_node)
        cur_wt = visited_dist[connected_node]
        for edge in graph.edges[connected_node]:
            wt = cur_wt + graph.distances[(connected_node, edge)]
            if edge not in visited_dist or wt < visited_dist[edge]:
                visited_dist[edge] = wt
    return visited_dist
    
```

*  Isomap 함수 생성

```python
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import KernelPCA
from dijkstra import Graph, dijkstra
import numpy as np
import pickle

def isomap(input, n_neighbors, n_components, n_jobs):
    distance_matrix = pickle.load(open('./isomap_distance_matrix.p', 'rb'))
    kernel_pca_ = KernelPCA(n_components=n_components,
                                 kernel="precomputed",
                                 eigen_solver='arpack', max_iter=None,
                                 n_jobs=n_jobs)
    Z = distance_matrix ** 2
    Z *= -0.5
    embedding = kernel_pca_.fit_transform(Z)
    return(embedding)
```

*  MNIST 데이터 활용하여 모델 적용 

```python
from sklearn import (datasets,random_projection)
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from isomap import isomap

digits = datasets.load_digits(n_class=7)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
n_neighbors = 30

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


n_img_per_row = 30
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')

rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits")

t0 = time()
embedding = isomap(input=X, n_neighbors=n_neighbors, n_components=2, n_jobs=4)
plot_embedding(embedding,"Isomap projection of the digits (time %.2fs)" %(time() - t0))

embedding_three_dim = isomap(input=X, n_neighbors=n_neighbors, n_components=3, n_jobs=4)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(embedding_three_dim[:, 0], embedding_three_dim[:, 1], embedding_three_dim[:, 2], c=y)

```

*  최종 시각화 결과 확인

총 7개의 숫자가 2차원에서도 잘 분류되었고 3차원에서도 각 색깔이 뚜렷하게 군집을 이루고 있음을 확인할 수 있습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/lpviFTG.png"></p>

<p align="center"> 그림5 </p>

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/J9Rdi7l.png"></p>

<p align="center"> 그림6 </p>

---

LLE
-------

로컬 선형 임베딩(Local Linear Embedding)은 고차원의 공간에서 인접해 있는 데이터들 사이의 선형적 구조를 보존하면서 저차원으로 임베딩하는 방법론입니다. 즉 좁은 범위에서 구축한 선형모델을 연결하면 다양체, 매니폴드를 잘 표현할 수 있다는 알고리즘입니다. LLE는 다음과 같은 장점을 갖습니다. 
1. 사용하기에 간단하다. 
2. 최적화가 국소최소점으로 가지 않는다.
3. 비선형 임베딩 생성이 가능하다.
4. 고차원의 데이터를 저차원의 데이터로 매핑이 가능하다.

LLE 알고리즘은 3 단계로 구성됩니다.

**1.	가장 가까운 이웃 검색**

각 데이터 포인트 점에서 k개의 이웃을 구합니다.  

**2.	가중치 매트릭스 구성**

현재의 데이터를 나머지 k개의 데이터의 가중치의 합을 뺄 때 최소가 되는 가중치 매트릭스를 구합니다. 

$$
 E(W) = \sum_i \left|x_i - \sum_j W_{ij} x_j\right|^2
$$


s.t. $ W_{ij} = 0 $ if $ x_j $ 가 $x_i$의 이웃에 속하지 않을때 모든 i 에 대하여 $\sum_j W_{ij} = 1$  

**3.	부분 고유치 분해**

앞서 구한 가중치를 최대한 보장하며 차원을 축소합니다. 이때 차원 축소된 후의 점을 Y로 표현하며 차원 축소된 $ Y_j $와의 값 차이를 최소화하는 Y를 찾습니다. 

$$
\Phi(W) = \sum_i \left| y_i - \sum_j W_ik y_j \right|^2
$$ 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/SBVKuSc.png"></p>

<p align="center"> 그림7 </p>


그림 7는 LLE학습 과정을 나타냈습니다. 가중치와 벡터는 비록 선형대수의 방법으로 계산되지만 점들이 이웃 점들에게서만 재구축된다는 조건은 비선형 임베딩 결과를 초래하기에 nonlinear mapping으로 간주됩니다. 
LLE의 계산복잡도는 아래와 같습니다.

$$ O[D \log(k) N \log(N)] + O[D N K^3] + O[d N^2] $$

+ N: 훈련 데이터 포인트의 수 
+ D: 입력 차원수
+ k: 가장 가까운 이웃의 수
+ d: 출력 차원수

### LLE을 활용한 예시 


*  1000개의 스위스롤을 구성하는 데이터를 2차원으로 차원 축소 

```python
import pylab as pl
import numpy as np

def swissroll():
    N = 1000
    noise = 0.05
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, N))
    h = 21 * np.random.rand(1, N)
    data = np.concatenate((t * np.cos(t), h, t * np.sin(t))) + noise * np.random.randn(3, N)
    return np.transpose(data), np.squeeze(t)

def LLE(data, nRedDim=2, K=12):
    ndata = np.shape(data)[0]
    ndim = np.shape(data)[1]
    d = np.zeros((ndata, ndata), dtype=float)
    for i in range(ndata):
        for j in range(i + 1, ndata):
            for k in range(ndim):
                d[i, j] += (data[i, k] - data[j, k]) ** 2
            d[i, j] = np.sqrt(d[i, j])
            d[j, i] = d[i, j]

    indices = d.argsort(axis=1)
    neighbours = indices[:, 1:K + 1]
    W = np.zeros((K, ndata), dtype=float)

    for i in range(ndata):
        Z = data[neighbours[i, :], :] - np.kron(np.ones((K, 1)), data[i, :])
        C = np.dot(Z, np.transpose(Z))
        C = C + np.identity(K) * 1e-3 * np.trace(C)
        W[:, i] = np.transpose(np.linalg.solve(C, np.ones((K, 1))))
        W[:, i] = W[:, i] / np.sum(W[:, i])

    M = np.eye(ndata, dtype=float)
    for i in range(ndata):
        w = np.transpose(np.ones((1, np.shape(W)[0])) * np.transpose(W[:, i]))
        j = neighbours[i, :]
        ww = np.dot(w, np.transpose(w))
        for k in range(K):
            M[i, j[k]] -= w[k]
            M[j[k], i] -= w[k]
            for l in range(K):
                M[j[k], j[l]] += ww[k, l]
    evals, evecs = np.linalg.eig(M)
    ind = np.argsort(evals)
    y = evecs[:, ind[1:nRedDim + 1]] * np.sqrt(ndata)
    return evals, evecs, y

data, t = swissroll()
evals, evecs, y = LLE(data)

t2= t.min()
t3= t.max()
t = (t-t2) / (t3-t2)
pl.scatter(y[:, 0], y[:, 1], s=50, c=t, cmap=pl.cm.gray)
pl.axis('off')
pl.show()

```
*  스위스롤 데이터의 차원 축소 시각화 결과 

검은색과 흰색으로 색깔에 따라 데이터의 특징을 잘 보존하며 차원 축소가 이루어 졌음을 확인할 수 있습니다. 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/ZXznmRR.png"></p>

<p align="center"> 그림8 </p>



---

t-SNE
-------

t-SNE 전에  SNE(Stochastic Neighbor Embedding)부터 설명하겠습니다.
SNE는 고차원 공간에서 유클리드 거리를 포인트들간의 유사성을 표현하는 조건부 확률로 변환하는 방법입니다. 두 점 i에 대해 j와의 유사도를 나타내는 조건부 확률은 i를 중심으로하는 가우시안 분포(정규 분포)의 밀도에 비례하여 근방이 선택되도록 하는 확률을 의미합니다. 즉 조건부 확률이 높다면 서로간의 유사성이 높아 포인트의 거리가 가깝고 반대일 경우에는 거리가 멀다고 해석할 수 있습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/7qATrbV.png"></p>

<p align="center"> 그림9 </p>


본격적으로 SNE알고리즘 계산을 위해 고차원 공간의 데이터 포인트간의 거리 정보를 보존하는 저차원 데이터 포인트를 정의합니다. 그림7은 고차원과 저차원에서 각 데이터 포인트끼리의 조건부 확률, 즉 유사도를 의미합니다.  
만일 고차원의 데이터 포인트끼리의 거리 정보가 저차원의 포인트간에서도 잘 보존되었다면 그림 10의 $p_{j|i} $ 와 $ q_{j|i} $ 가 유사할 것입니다. 두 확률 분포의 유사도를 측정하는 지표로 KL-divergence(Kullback-Leibler divergence)가 있습니다. 최소 0에서 1까지의 값을 가지며 동일할수록 그 값이 낮습니다. 즉 그림11과 같이 모든 데이터 포인트에 대해서 KL divergence값의 총합을 최소화 하는 방향으로 학습이 진행되며 최소화는 gradient descent를 통해 수행됩니다. 

고차원 데이터 포인트 | 저차원 데이터 포인트 
------ | ------
 $ x_i, x_j$ |  $ y_i, y_j $ 

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/9By9iXh.png"></p>

<p align="center"> 그림10 </p>

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/TSohbs1.png"></p>

<p align="center"> 그림11 </p>




SNE는 앞서 언급했듯이 가우시안 분포를 가정합니다. 그런데 가우시안 분포는 양쪽 꼬리가 충분히 두텁지 않습니다. 즉 일정 거리 이상부터는 아주 멀리 떨어져 있어도 선택될 확률이 큰 차이가 나지 않게되는데 이를 Crowding Problem이라고 합니다. 이 단점을 완화하기 위해 가우시안 분포와 유사하지만 좀 더 양 끝이 두터운 자유도 1의 t분포를 사용합니다. 이것이 바로 t-SNE입니다. 그림12는 가우시안 분포와 t분포의 차이를 보여줍니다. SNE의 $p_{ij}$는 동일하게 사용하며 대신 $q_{ij}$에만 t분포를 적용합니다. t분포를 적용한 $q_{ij}$ 는 아래와 같습니다. 


<p align="center"><img width="500" height="auto" src="https://i.imgur.com/FYRDh4p.png"></p>

<p align="center"> 그림12 </p>




t-SNE의 장점은 PCA와는 달리 군집이 증복되지 않는다는 점입니다. 그렇기에 시각화에 굉장히 유용합니다. 또한 지역적인 구조를 잘 잡아내는 동시에 글로벌적 특징도 놓치지 않음이 알려져 있습니다. 아래 그림 13과 같이 각 숫자별 클러스터가 잘 형성되며 동시에 유사한 모습의 숫자인 7과 9의 위치가 굉장히 가까이 나타남을 확인할 수 있습니다.

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/ho78NYk.png"></p>

<p align="center"> 그림13 </p>

반면에 매 시도마다 임의로 데이터 포인트를 선정하기에 축의 위치가 계속해서 변해 모델의 학습용으로는 좋지 않습니다. 또한 계산 비용이 많이 들어 학습이 오래걸립니다. 같은 데이터에서도 PCA에 비해 크게 긴 계산 시간을 요구합니다.  



### t-SNE를 활용한 예시

t-SNE를 활용하여 5개의 MNIST 글씨 데이터 2차원 축소 시각화

*  t-SNE 학습을 위한 함수들 생성

```python
import numpy as np

def neg_distance(X):
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D
def softmax(X):
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
    np.fill_diagonal(e_x, 0.)
    e_x = e_x + 1e-8
    return e_x / e_x.sum(axis=1).reshape([-1, 1])
def calc_prob_matrix(distances, sigmas=None):
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)
def _perplexity(prob_matrix):
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity
def perplexity(distances, sigmas):
    return _perplexity(calc_prob_matrix(distances, sigmas))
def binary_search(fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess
def find_optimal_sigmas(distances, target_perplexity):
    sigmas = []
    for i in range(distances.shape[0]):
        fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        correct_sigma = binary_search(fn, target_perplexity)
        sigmas.append(correct_sigma)
    return np.array(sigmas)
def p_conditional_to_joint(P):
    return (P + P.T) / (2. * P.shape[0])
def q_joint(Y):
    inv_distances = neg_squared_euc_dists(Y)
    distances = np.power(1., -inv_distances,-1)
    np.fill_diagonal(distances, 0.)
    return distances / np.sum(distances), distances
def p_joint(X, target_perplexity):
    distances = neg_distance(X)
    sigmas = find_optimal_sigmas(distances, target_perplexity)
    p_conditional = calc_prob_matrix(distances, sigmas)
    P = p_conditional_to_joint(p_conditional)
    return P
def tsne_grad(P, Q, Y, distances):
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    distances_expanded = np.expand_dims(distances, 2)
    y_diffs_wt = y_diffs * distances_expanded
    grad = 4. * (pq_expanded * y_diffs_wt).sum(1)
    return grad
def t_SNE(X, num_component, num_iters = 500, learning_rate = 10., momentum = .9):
    Y = np.random.normal(0., 0.0001,[X.shape[0], num_component])
    P = p_joint(X, 20)
    if momentum:
        Y_m2 = Y
        Y_m1 = Y
    for i in range(num_iters):
        Q, distances = q_joint(Y)
        grads = tsne_grad(P, Q, Y, distances)
        Y = Y - learning_rate * grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
            Y_m2 = Y_m1
            Y_m1 = Y
    return Y
```


*  MNIST Datasets을 활용한 t-SNE 학습


```python
from sklearn import (datasets)
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from tsne import t_SNE

digits = datasets.load_digits(n_class=5)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
print("Computing t_SNE embedding")
t0 = time()
embedding = t_SNE(X = X, num_component = 2)

plot_embedding(embedding,"t_SNE projection of the digits (time %.2fs)" %(time() - t0))
```

*  학습된 데이터를 활용한 시각화 결과 

각 숫자별 분류가 잘 되었고 비슷한 모양의 '2'와 '3'이 인접함을 알 수 있습니다.

<p align="center"><img width="500" height="auto" src="https://i.imgur.com/6Wz4nNi.png"></p>

<p align="center"> 그림14 </p>
