---
title: numpy1
date: 2018-08-18 17:09:16
categories:
- Python
tags:
- numpy
- python
---


# 1 numpy

numpy的一些使用

## 1.1 random函数的使用

主要有一些一些函数的使用
- normal()
- random()

### 1.1.1 normal

numpy.random.normal(loc=0.0, scale=1.0, size=None)
从正态分布中抽取随机样本
- loc 分布的平均值
- scale 标准偏差
- size 整数或者元组，比如(m,n,k),那么得到的就是m行，n列，k高的三维数组列表


```python
import numpy as np

np.random.normal(0, 0.5, (3,4))
```




    array([[ 0.06158366,  0.94496841, -0.42095387, -0.16682898],
           [-0.3145212 , -0.41039232,  0.12504023, -0.17090629],
           [ 0.32857722, -0.25585145, -0.1607699 ,  0.228075  ]])



### 1.1.2 random

numpy.random.random(size=None) ，返回0-1间的数（不包含1），size的解释和上文一样，如果size=None,那么默认就是随机产生一个0-1间的数字


```python
np.random.random((3,4))
```




    array([[0.08519073, 0.40447468, 0.92314145, 0.11433058],
           [0.08232515, 0.89703947, 0.86035142, 0.7718275 ],
           [0.69032194, 0.88780312, 0.92649113, 0.72175598]])



## 1.2 数组的转置

### 1.2.1 一维数组的转置  reshape

reshape()  numpy.reshape(a, newshape, order='C')
- a 表示原始的要整型的数组
- newshape，可以是一个整数或者一个元组。比如转换成一列，可以使用(-1,1),转换成一行可以使用(1,-1)   ,-1表示不确定多少行（列）
- order暂时用不上，没有了解


```python
a = np.array([[1,2,3],[4,5,6]])
a = a.reshape(-1,1)
a

b = np.reshape(a, (-1,1))
b
# 使用a.T 无效
```




    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])



## 1.3 基本运算

包括
- sum
- max min
- mean
- 矩阵乘法

### 1.3.1 总体上


```python
a = np.array([[1,2,3],[4,5,6]])
# 计算所有项的和
a.sum()

# 计算最大值,最小值，均值
a.max()
a.min()
a.mean()
```




    3.5



### 1.3.2 行或者列上


```python
# 计算每一行的和
a.sum(axis = 1)
a.max(axis=1)
```




    array([3, 6])



### 1.3.3 矩阵乘法


```python
a = np.array([1,2,3])
b = np.array([[1],[2],[3]])
# 矩阵乘法  dot
a.dot(b)    
np.dot(a,b)

# 对应位置相乘，不清楚为什么矩阵的行列不相等的时候也可以进行乘法运算
a*b


```




    array([[1, 2, 3],
           [2, 4, 6],
           [3, 6, 9]])


