---
layout: post
comments: true
title: OverFeat 부터 Mask R-CNN까지 (1)
categories: Object Detection
tags:
- Object Detection
---

안녕하세요 오늘은 OverFeat:Integrated Recognition, Localization and Detection using Convolutional Networks 논문을 살펴보겠습니다. 본 논문은 2013년도의 논문으로 시기가 좀 지났지만
워낙 유명한 논문이기에 한글로도 이 논문에 대해 다룬 포스트를 찾아볼 수 있습니다. 본 포스트는 https://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220752877630&parentCategoryNo=&categoryNo=22&viewDate=&isShowPopularPosts=false&from=postView , http://dhhwang89.tistory.com/135 와 고려대학교 DSBA 연구실의 천우진 석사과정의 발표자료를 종합하여 작성하였습니다. 

OverFeat의 가장 큰 특징은 Classification, Localization, Detection을 하나의 프레임워크로 통합하였다는 점에 있습니다. 
