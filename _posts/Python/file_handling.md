---
layout: post
comments: true
title: Python_File_Handling
categories: Python
tags:
- Python
- glob

---

Python File Handling
=======

안녕하세요 이번 포스트는 Python에서 파일 접근법을 다루도록 하겠습니다. 대표적인 라이브러리로 Glob이 있습니다. 

To begin
-------

Glob은 파일들의 리스트를 뽑을 때 사용합니다. Os.listdir과 유사하지만 경로명을 사용할 수 있다는 장점이 있습니다. 

```python
from glob import glob

glob('*.jpg')

>> ['abc.jpg', 'efg.jpg']
```


Combine
-------

Glob의 특징을 활용하여 os.path와 결합하면 사용하고자하는 데이터를 유연하게 호출할 수 있게됩니다. 

```python
from glob import glob
import os

glob(os.path.join(os.getcwd(),'train','*.jpg')

>> ['tr_0.jpg', 'tr_1.jpg', 'tr_2.jpg']
```

