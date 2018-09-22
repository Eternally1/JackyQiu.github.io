---
title: matplotlib
date: 2018-09-08 09:03:50
categories:
- Python
tags: ["Python","matplotlib"]
---
matplotlib画图中的一些方法的使用

## 1.1 scatter 散点图

matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)  
画y与x的散点图，可以标记成不同的颜色或者大小  
- x，y就分别代表数据，是数组类型
- s 标记的大小 以平方磅为单位的标记面积，指定为下列形式之一：
    - 数值标量 ： 以相同的大小绘制所有标记。
    - 行或列向量 ： 使每个标记具有不同的大小。x、y 和 sz 中的相应元素确定每个标记的位置和面积。sz 的长度必须等于 x 和 y 的长度。
    - [] ： 使用 36 平方磅的默认面积。
    但是行和列向量，[]这两种如何使用？
- c 颜色，其中对应如下  b=blue  c=cyan g=green k=black m=magenta r=red w=white y=yellow
- marker 标记样式 'o' 圆圈   '+'＋号  等等

一些具体的使用案例可以参考本博客的[聚类](https://eternally1.github.io/2018/09/13/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8E%E6%95%B0%E6%8D%AE%E6%8C%96%E6%8E%98/%E8%81%9A%E7%B1%BB/)

## 1.2 一些小的方法

### 1.2.1 figure & subplots

plt.figure()用来产生多个图，图片按照顺序增加，需要注意一个概念，当前图和当前坐标。通过下面一个例子了解figure和subplot的使用方法。


```python
import matplotlib.pyplot as plt
plt.figure(1)                # 第一张图
plt.subplot(211)             # 第一张图中的第一张子图
plt.plot([1,2,3])
plt.subplot(212)             # 第一张图中的第二张子图
plt.plot([4,5,6])


plt.figure(2)                # 第二张图
plt.plot([4,5,6])            # 默认创建子图subplot(111)

plt.figure(1)                # 切换到figure 1 ; 子图subplot(212)仍旧是当前图
plt.subplot(211)             # 令子图subplot(211)成为figure1的当前图
plt.title('Easy as 1,2,3')   # 添加subplot 211 的标题
plt.show()
```

#### subplots
    matplotlib.pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
    比较常用的参数也就是前面两个，代表创建的子图的行列数量。返回值是fig和ax，分别表示Figure和Axes的对象，可以看1.5中的matplotlib的详细构造。

一些小的例子，只创建一个axes对象，此时在subplots默认为创建一个Figure和一个axes
```python
import matplotlib.pyplot as plt

fig,ax = plt.subplots()
ax.plot([1,2])
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([1,2,3])
ax2.plot([1,2,3])
```

```python
fig, axes = plt.subplots(2, 2)
axes[0,0].plot([1,2,3])
axes[1,1].plot([1,2,3])
```
<img src="/images/Python/matplotlib/subplots_01.png" width="400px" height="300px">


    
### 1.2.2 plt.xlabel  plt.ylabel等

xlabel   ylabel分别是用来设置坐标系的名称的。

### 1.2.3 图例legend

调用matplotlib.pyplot.legend(*args, \**kwargs)可以设置legend,有如下一些参数：
- loc   可选值：best，upper right ， upper left，lower right|left， center right|left, lower center .etc
- fontsize 字体大小，可以设置int或者xx-small x-small small medium large x-large xx-large
- fremeon  False or True  设置图例边框
- edgecolor 设置图例边框颜色  blue  red etc.
- facecolor  设置图例背景颜色，若无边框，参数无效
- title 设置图例标题

### 1.2.4 annotate 添加注释


```python
# 添加注释
# 第一个参数是注释的内容
# xy设置箭头尖的坐标
# xytext设置注释内容显示的起始位置
# arrowprops 用来设置箭头
# facecolor 设置箭头的颜色
# headlength 箭头的头的长度
# headwidth 箭头的宽度
# width 箭身的宽度
plt.annotate(u"This is a zhushi", xy = (0, 1), xytext = (-4, 50),
             arrowprops = dict(facecolor = "r", headlength = 10, headwidth = 30, width = 20))

```

### 1.2.5解决中文显示问题


```python
# solve the problem that the axis can not display chinese normally
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
```

### 1.2.6 设置坐标轴的可读xticks()

这个之后在进行补充。

### 1.2.7 %matplotlib inline
经常会看到这么一行代码  
%matplotlib inline  

但是放到自己的IDE环境下运行时，总是报错。
在Stack Overflow上看到了一个解释:
IPython有一组预定义的“魔术函数”，您可以使用命令行样式语法调用它们。有两种魔法，一种是线导向（line-oriented），另一种是单元导向(cell-oriented)。line magics以%字符作为前缀，其工作方式与操作系统命令行调用非常相似:它们作用于整行,line magics可以返回结果，也可以进行赋值使用；cell magics是以%%开头，它需要出现在单元的第一行，而且是作用于整个单元。  

使用此方法时，绘制命令的输出将在前端显示，就像Jupyter笔记本一样，直接显示在生成命令的代码单元格的下方，生成的绘图也将存储在笔记本文档中。不过这个方法好像只适用于Jupyter notebook和Jupyter QtConsole。

### 1.2.8 rcParams
matplotlib.rcParams[‘figure.figsize’]  # 图片尺寸     
plt.rcParams['figure.figsize'] = (16, 9)   # 代表16:9这种大小的。或者4:3等 


### 1.2.9 style.use
plt.style.use("ggplot")使用自带的样式进行美化  
plt.style.available  获取所有的自带样式  
可以对生成的图形进行美化  

## 1.3 plot 折线图

plot([x], y, [fmt], data=None,\**kwargs)
fmt = '[color][marker][line]'设置这个属性可以分别设置颜色，点的样式以及连线的样式。  
\**kwargs中可以是包含关键词的属性：  
- label   用于自动图例的


```python
time= ["07-22","07-23","07-24","07-25","07-26","07-27","07-28"]

counts_baidu = [13879,17886,11305,5961,3086,2331,1680]
counts_weibo = [ 14201,19541,9084,6014,4102,1900,1702 ]

plt.figure(1)
plt.xlabel('time')
plt.ylabel('counts')


plt.plot(time, counts_baidu, 'b.-', label="baidu")
plt.plot(time,counts_weibo,'g.--',label="weibo")

# 设置legend,前提是需要在plot中添加label属性。
plt.legend(loc="upper right",fontsize=12,facecolor='white')

plt.show()
```


## 1.4 bar  柱状图

matplotlib.pyplot.bar(*args, \**kwargs)
参数有如下：
       width: 设置柱形条宽度，同时也就设置了柱形条之间的宽度
       


```python
import matplotlib.pyplot as plt

# solve the problem that the axis can not display chinese normally
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

top_10_nouns = [('疫苗', 1405), ('长生', 866), ('生物', 455), ('问题', 311), ('事件', 276), ('狂犬病', 176), ('公司', 149), ('记录', 143), ('企业', 125), ('狂犬', 89)]

plt.xlabel(u'关键词')
plt.ylabel('次数')

# 1、determine the value of x and y axis
x = [item[0] for item in top_10_nouns]
y = [item[1] for item in top_10_nouns]

# 2 use bar function to draw the graph
plt.bar(x, y,width=0.6,color='b')

```
<img src="/images/Python/matplotlib/bar_01.png" width="400px" height="300px">

```python
import matplotlib.pyplot as plt

# solve the problem that the axis can not display chinese normally
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

top_10_nouns = [('疫苗', 1405), ('长生', 866), ('生物', 455), ('问题', 311), ('事件', 276), ('狂犬病', 176), ('公司', 149), ('记录', 143), ('企业', 125), ('狂犬', 89)]
top_10_verbs = [('造假', 233), ('接种', 152), ('有', 108), ('相关', 74), ('是否', 70), ('关注', 66), ('能', 52), ('称', 52), ('采购', 52), ('涉事', 45)]

# for set xlabels
fig = plt.figure()
ax = fig.add_subplot(111)

plt.xlabel(u'关键词')
plt.ylabel('次数')

# 1、determine the value of x and y axis
x1 =[1,2,3,4,5,6,7,8,9,10]
labels = [item[0] for item in top_10_nouns]
y1 = [item[1] for item in top_10_nouns]

# 2 use bar function to draw the graph
plt.bar(x1, y1,width=0.2,color='b',lw=1)
ax.set_xticklabels(labels)


x2 = [item+0.2 for item in x1]
y2 = [item[1] for item in top_10_verbs]


plt.bar(x2, y2,width=0.2,color='r',lw=1)
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.show()

```
<img src="/images/Python/matplotlib/bar_02.png" width="400px" height="300px">

## 1.5 关于matplotlib各个图像部分的解释
可以参考[这里](https://www.cnblogs.com/nju2014/p/5620776.html),主要是关于figure，axes，subplot之间的关系进行了说明，比较清楚。

首先一幅Matplotlib的图像组成部分介绍。

在matplotlib中，整个图像为一个Figure对象。在Figure对象中可以包含一个或者多个Axes对象。每个Axes(ax)对象都是一个拥有自己坐标系统的绘图区域。所属关系如下：
<img src="/images/Python/matplotlib/figure_01.png" width="400px" height="300px">

这里有一个系列的matplotlib的教程，可以增加理解。[here](https://www.cnblogs.com/nju2014/tag/Matplotlib/)

