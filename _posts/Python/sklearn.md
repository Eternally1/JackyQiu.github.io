---
title: sklearn
date: 2018-09-08 09:03:50
categories:
- Python
tags: ["Python","matplotlib"]
---
# 引言

sklearn机器学习包中的一些方法说明和使用

## 1.1 train_test_split

使用scitkie-learn中的一个方法，可以随机生成测试集和训练集。
sklearn.model_selection.train_test_split(train_data, train_target, test_size, train_size, random_state)
- test_size 默认情况下为0.25,可以为浮点数或者int型，在一致train_size的时候，它就确定了
- random_state 随机数种子，其实就是改组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如每次都填1，其他参数设置一样的情况下得到的随机数组是一样的，但是如果填0或者不填，每次都会不一样。随机数的产生取决于中农资；随机数和种子之间的关系遵从以下两个原则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

返回值跟传入的数组有关，如果有x，y，返回的就是2*len(x,y) = 2*x = 4。


```python
# train_test_split是从model_selection中导入的。
import numpy as np
from sklearn.model_selection import train_test_split
X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
X
```




    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]])



## 1.2 SVC()

class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
官方文档：
c-支持向量分类：该实现是基于libsvm，拟合事件复杂度大于样本数量的平方，因此当样本数量大于10000的时候就不适用了。同时它的多类支持是通过一对一的方案解决的。
[全面的支持向量机文档](http://scikit-learn.org/stable/modules/svm.html#svm-classification)

- C 惩罚参数
- kernel 内核函数，可以是linear  poly  rbf  sigmoid  precomputed 分别是线性核函数、多项式核函数（最常用的径向机核函数就是高斯核函数）、径向机核函数、神经元的非线性作用函数核函数、用户自定义核函数。
- degree 当使用多项式核函数的时候，degree定义了该核函数的多项式次数。
- gamma rbf  poly  sigmoid的核系数。【具体含义不清楚】
- decision_function_shape   ovr就是one vs rest  一个类别与其他类别进行划分     ovo  one vs one类别两两之间进行划分，即使用二分类的方法模拟多分类的结果。

还包括一些训练完成之后的方法，碧土score  predict   decision_function等比较简单，可以查看官方文档。
