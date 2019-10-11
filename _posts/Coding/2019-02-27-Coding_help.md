---
layout: post
comments: true
title: Python coding
categories: Python
tags:
- Python

---

Python coding help 
=======



안녕하세요. 고려대학교 DSBA연구실 석사과정 양우식입니다.


이번 포스트에서는 파이썬에서 도움이 되는, 기억할만한 팁 및 유용한 모듈들을 모아 봤습니다. 
제가 스스로 공부하면서 느꼈던 유용한 부분을 모아 기록한건데요, 이번 포스트는 생각나는 팁이 생길 때마다 추가하는 형식입니다.  


- 1. 그래프 시각화 모듈 Graphiz

노드와 엣지로 표현해주는 시각화에 좋은 패키지입니다. 
 
https://graphviz.readthedocs.io/en/stable/ 로 가시면 상세한 설명을 볼 수 있습니다. 


- 2. 현재 설치된 패키지 확인 

다들 아시겠지만 굉장히 유용한 패키지입니다. 

freeze 패키지로 설치된 패키지와 버전을 확인할 수 있습니다. 

- 3. 파일 directory 알기 

import os
print (os.getcwd()) #현재 디렉토리의
print (os.path.realpath(__file__))#파일
print (os.path.dirname(os.path.realpath(__file__)) )#파일이 위치한 디렉토리

getcwd() : 현재작업디렉토리(current working directory)를 나타내는 스트링을 리턴
__file__ : 파일의 path를 저장
realpath() : file이 symbolic link인 경우 원본 위치를 찾아줌


출처 : https://hashcode.co.kr/questions/197/python%EC%9C%BC%EB%A1%9C-%ED%98%84%EC%9E%AC-%EB%94%94%EB%A0%89%ED%86%A0%EB%A6%AC-%EC%9C%84%EC%B9%98%EB%A5%BC-%EC%95%8C%EC%95%84%EB%82%B4%EB%8A%94-%EB%B0%A9%EB%B2%95


