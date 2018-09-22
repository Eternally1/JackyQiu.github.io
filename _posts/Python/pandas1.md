---
title: pandas1
date: 2018-08-18 17:08:48
categories:
- Python
tags:
- Python
- pandas
---

# 1 pandas学习使用

# 1.1 一些基本的操作

包括
- 文件读取 统计信息，部分数据查看，绘制相关图形。

### 1.1.1 文件读取


```python
import pandas as pd
filename = "C:/Users/14259/Desktop/25周/watermelon.csv"
dataset = pd.read_csv(open(filename), delimiter=",")

#  显示一些统计信息
dataset.describe()

# 显示前几条记录显示后面几条记录使用tail
dataset.head()

# 绘制图形，参数是数据集的某列。
dataset.hist("密度")
```

一些使用案例可以参考本博客支持向量机的2.1章节[here](https://eternally1.github.io/2018/08/23/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98/%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/)

## 1.2 关于列和行的操作


包括
- loc  at  index  shape  columns  dtypes

### 1.2.1 一些基础的操作  loc  at  index  columns  shape等


```python
df = dataset
df['密度']    # 获取columns为“密度”的列
df[1:3]    # get the first and second lines
df.loc[1:4,'编号':'纹理']   #行切片加列切片

df.at[4,'色泽']    # 获取指定位置的元素，第4行，色泽这一列

df.dtypes

df.index

# 查看列索引
df.columns

# 每一行作为列表的一项进行返回
df.values

# 每一列作为列表的一项进行返回
df = df.T
df.values

df.shape

```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]



### 1.2.2 删除指定索引列


```python
df = df.T
x = [1,2]
df.drop(df.columns[x], axis=1, inplace=True)
df

# 删除指定的索引行，注意axis=0
df.drop(df.index[x], axis=0, inplace=True)
df
```



