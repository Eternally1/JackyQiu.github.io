---
title: 标准BP算法Python实现
date: 2018-08-15 17:11:51
categories: 
- 数据挖掘
tags: ["BP算法","算法"]
toc: true
comments: true
---

# 引言

根据机器学习-周志华书上讲解的BP算法的思想，使用西瓜数据集进行代码实现

# 1 数据集

西瓜数据集内容如下图所示：
<img src="/images/DataMiningTheory/BP/BP_01.png" width="600px" height="100px" />

接着是导入数据集并对数据集进行分析和处理


```python
import csv
import math
import collections
import copy
filename = r"C:\Users\14259\Desktop\25周\watermelon.csv"
```


```python
def loadDataset(filename):
        csv_reader = csv.reader(open(filename));
        dataset = list(csv_reader);
        return dataset[1:]
dataset = loadDataset(filename)
```

因为数据集中既包含连续变量，也包含离散变量。需要对离散属性进行处理，如果属性值之间存在“序”的关系则可以进行连续化；否则通常转化为K维向量，k为属性值数。参考本书的3.2节。


```python
# 通过代码实现将离散变量转化成k维向量
def getSingleAttrVector(attr_list):
    """将单列属性的值进行转换成向量，attr_list是属性列表
    """
    attrs = list(set(attr_list));  # 属性取值集合
    attr_len = len(attrs);    # 属性取值的个数
    attr_map = collections.defaultdict()
    
    for i in range(len(attrs)):
        temp ="{0:0{attr_len}b}".format(int(math.pow(2,i)),attr_len = attr_len)   # 得到对应属性的二进制字符串
        # 对字符串进行切割，形成列表
        attr_map[attrs[i]] =[int(i) for i in tuple(temp)]
        
    attr_lists = [];
    for i in range(len(attr_list)):
        attr_lists.append(attr_map[attr_list[i]])
    return attr_map,attr_lists    # 得到对应的向量编码对应列表和相连编码

# attr_list = list(zip(*dataset))[1];
# getSingleAttrVector(attr_list)
```


```python
# 将数据集中所有的离散变量进行编码
def getAllAttrVector(dataset, discrete_index):
    """ dataset数据集，discrete_index离散属性列所在的索引。
    """
    dataset_copy = list(zip(* copy.deepcopy(dataset)));
    for i in discrete_index:
        attr_list = dataset_copy[i];
        dataset_copy[i] = getSingleAttrVector(attr_list)[1];
    return list(zip(*dataset_copy))

```

## 1.1 使用numpy pandas 数据预处理


```python
import pandas as pd
import numpy as np
filename = "C:/Users/14259/Desktop/25周/watermelon.csv"
```

考虑到离散特征的取值之间没有大小的意义，可以使用ont-hot编码。如果离散数值的取值有大小的意义，就使用数值进行映射。
[ont-hot](https://blog.csdn.net/pipisorry/article/details/61193868)


```python
dataset = pd.read_csv(open(filename), delimiter=",")

def preprocessData(dataset):
    
    # 删除编号列
    x = [0]   
    dataset.drop(dataset.columns[x], axis=1,inplace=True)
    
    #  对标签列进行映射
    temp = [];
    for i in range(len(dataset.iloc[:,-1])):
        if dataset.iloc[:,-1][i].strip() == '是':
             temp.append(1)
        else:
             temp.append(0);
    dataset.iloc[:,-1] = temp
    
#     print(type(dataset))
    
    # 将lable标签单独删除并保存
    labels = dataset[dataset.columns[-1]]
#     print(labels)
    
#  删除标签列
    x = [8]   
    dataset.drop(dataset.columns[x], axis=1,inplace=True)
#     print(dataset)
    columns = dataset.columns[0:6];
    
    # 对指定列进行ont-hot编码
    dataset = pd.get_dummies(dataset, columns=columns);
    return dataset,labels;
# dataset,labels = preprocessData(dataset)
# 划分成训练集和验证集
# trainSet,validSet = dataset[:15],dataset[15:]
# trainSet
```

# 2 代码实现

## 2.1 定义各种参数


```python
input_nodes = dataset.shape[1]
output_nodes = 1
hidden_nodes = 10    # 隐藏层个数

learning_rate = 0.001
epochs = 1000     # 迭代次数
batch_size = 1     # 每一次处理数据的大小
```

## 2.2 激活函数


```python
def sigmoid(inx):
    return 1.0/(1.0+math.exp(-inx))
```

## 2.3 随机生成权重和阈值

在0-1范围内初始化所有连接权和阈值


```python
# 随机生成输入层到隐藏层，隐藏层到输出层之间的权重，还有隐藏层神经元和输出层神经元的阈值
def getWeights():
    """np.random.normal方法的使用可以参考官方文档，最后一个参数就是制定size的，如果以元组的形式，那么得到的size就是元组内的数值乘积"""
    # 产生0-1内的随机数，同时注意对应的数组形式。
    weights_input_hidden = np.random.sample((input_nodes,hidden_nodes))
    weights_hidden_output = np.random.sample( (hidden_nodes, output_nodes))
    
    threshold_hidden = np.random.random(hidden_nodes)
    threshold_output = np.random.random(output_nodes)
    
    return weights_input_hidden,weights_hidden_output,threshold_hidden, threshold_output
```

## 2.4 计算输出

根据输入参数和权重的乘积和值，得到当前的输出。需要注意的是：关于矩阵的乘法，注意对应的矩阵的行列数。


```python
def calculateOutput(x):
    """返回值是隐藏层输出和输出层输出"""
    print(x)
    b = [];
    # 遍历隐藏层结点，得到每一个隐藏层的输出
    for h in range(hidden_nodes):
        temp = 0;
        # 针对该隐藏层结点，计算所有输入节点加权和
        for i in range(input_nodes):
            # 输入层第i个神经元与隐藏层第h个神经元之间加权和，得到第h个神经元接收到的输入
            temp += weights_input_hidden[i][h]*x[i]
        
        # b[h]为第h个神经元的输出，使用sigmoid函数,求接收到的输入与该神经元阈值的差值的sigmoid函数。
        b.append(sigmoid(temp-threshold_hidden[h]))

        
    y = [];
    # 遍历输出结点
    for o in range(output_nodes):
        temp = 0;
        for h in range(hidden_nodes):
            # 使用隐藏层神经元h与输出层神经元o之间的权重与隐藏层h的输出b[h]
            temp += weights_hidden_output[h][o]*b[h];
            
        y.append(sigmoid(temp-threshold_output[o]))
  
    
    return b,y       
```

## 2.5 更新参数


```python
def getGandE(x,label,b,y):
    """x是当前数据"""
#     b,y = calculateOutput(x)
    # 遍历输出结点，针对每一个预测输出进行计算tmp，也就是书本上的公式5.10
    g = [];  
    for j in range(output_nodes):
        # 最后一个因子是本来的输出与神经网络输出的差值
        temp = y[j]*(1-y[j])*(label-y[j])
        g.append(temp)
    
#     print(g)
    
    # 书本上公式5.15
    e = []
    for h in range(hidden_nodes):
        tmp = 0
        for j in range(output_nodes):
            # 隐藏层与输出层之间的权重与g的乘积进行累加。
            temp += weights_hidden_output[h][j]*g[j]
        e.append( b[h]*(1-b[h])*temp)
    
    #更新隐藏层和输出层之间的权重
    for h in range(hidden_nodes):
        for j in range(output_nodes):
            weights_hidden_output[h][j] += learning_rate*g[j]*b[h]
    
    # 更新输出层的阈值
    for  j in range(output_nodes):
        threshold_output[j] += -learning_rate*g[j]
        
    # 更新输入层与隐藏层之间的权重
    for i in range(input_nodes):
        for h in range(hidden_nodes):
            weights_input_hidden[i][h] += learning_rate*e[h]*x[i]
            
    # 更新隐藏层的阈值
    for h in range(hidden_nodes):
        threshold_hidden[h]+= -learning_rate*e[h]
    
# getGandE(x,1)
# print(weights_hidden_output)
# print(threshold_output)
# print(weights_input_hidden)
# print(threshold_hidden)

```

## 2.6 开始训练    

```python
def train():
    print("train neural networks")
    # 1、获取训练数据集
    dataset = pd.read_csv(open(filename), delimiter=",")
    dataset,labels = preprocessData(dataset)
    trainSet,validSet = dataset[:15],dataset[15:]
    trainLabel,validLabel = labels[:15],labels[15:]
    
    # 2、初始化权重和阈值
    weights_input_hidden, weights_hidden_output, threshold_hidden,  threshold_output =  getWeights()
    
    trainSet

#     print(trainSet[1:2])
    epoch = 0
    # 3、开始循环
    while epoch<10:
        for i in range(len(trainSet)):
            pass
            x = np.array(trainSet[i:i+1])
            label = trainLabel[i]
            # 3.1 计算输出
            b,y = calculateOutput(x)
            # 3.2 计算g，e并更新参数
            getGandE(x,label,b,y)
        epoch+=1
```

## 2.7 计算误差

标准BP算法每次更新只针对单个样例，参数更新的非常频繁，而且对不同样例进行更新的效果可能出现“抵消“现象。因此，为了达到同样的累积误差极小值，标准BP算法往往需要进行更多次数的迭代。


```python
# 计算单个数据的误差
def calculateLoss(y,label):
    loss = 0;
    for j in range(output_nodes):
        loss += (y[j]-label[j])**2
    return loss


# 计算所有数据的误差
def calculateAllLoss(dataset,y,labels):
    for k in range (len(dataset)):
        loss = calculateLoss(y[k],labels[k])
        lossAll.append(loss)
    
    self.lossAverage = sum(lossAll)/len(dataset)
```

# 3 合并代码


将以上代码整合起来，写成一个类。合并之后的代码如下，英文注释是重写代码之后加上的，中文注释是上文写代码时使用的。


```python
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

FILENAME = "C:/Users/14259/Desktop/25周/watermelon.csv"
OUTPUT_NODES = 1
HIDDEN_NODES = 10
LEARNING_RATE = 0.001

class BP(object):
    """
    标准BP算法
    """
    def __init__(self, hidden_nodes, output_nodes, learning_rate, filename, input_nodes=None):
        """

        :param hidden_nodes:
        :param output_nodes:
        :param learning_rate:
        :param filename: the path of dataset
        :param input_nods: 输入节点需要在知道数据集的情况下确定，因此初始化的时候可以不用管
        """
        self.options = {}
        self.options['input_nodes'] = input_nodes
        self.options['output_nodes'] = output_nodes
        self.options['hidden_nodes'] = hidden_nodes
        self.options['learning_rate'] = learning_rate
        self.dataset = pd.read_csv(open(filename), delimiter=",")    # the raw data

        # the train dataset and validation dataset
        self.data = {}
        self.label = {}
        self.data['train'] = None
        self.data['valid'] = None   # the validation set
        self.label['train'] = None
        self.label['valid'] = None

        # the weights and threshold
        self.weights = {}
        self.threshold = {}
        self.weights['input_hidden'] = None
        self.weights['hidden_output'] = None
        self.threshold['hidden'] = None
        self.threshold['output'] = None

        # the output value of neural
        self.output = {}
        self.output['b'] = None  # the hidden layer output
        self.output['y'] = None  # the output layer output

        # the gradient of output neuron and hidden neuron
        self.gradient = {}
        self.gradient['output'] = None
        self.gradient['hidden'] = None

    def _preprocess_data(self):
        """
        数据预处理，
        :return:
        """
        ds = self.dataset

        # delete the useless attr columns, the number column here.
        x = [0]
        ds.drop(ds.columns[x], axis=1, inplace=True)

        # transfer the chinese into 1 or 0 for programming
        temp = []
        for i in range(len(ds.iloc[:, -1])):
            # ds.iloc[:, -1]is the method to get the last column
            if ds.iloc[:, -1][i].strip() == '是':
                temp.append(1)
            else:
                temp.append(0)
        ds.iloc[:, -1] = temp

        # get the labels column
        labels = ds[ds.columns[-1]]

        # delete the lables column
        x = [8]
        ds.drop(ds.columns[x], axis=1, inplace=True)
        columns = ds.columns[0:6];

        # use the ont-hot to encoding the dataset
        ds = pd.get_dummies(ds, columns=columns);

        # set the number of input node
        self.options['input_nodes'] = ds.shape[1]

        return ds, labels

    def split_dataset(self):
        """
        split the dataset into training dataset and validation dataset
        :return: None
        """
        ds, labels = self._preprocess_data()
        train_set, valid_set = ds[:12], ds[12:]
        train_label, valid_label = labels[:12], labels[12:]

        self.data['train'] = train_set
        self.data['valid'] = valid_set
        self.label['train'] = train_label
        self.label['valid'] = valid_label.reset_index(drop=True)  # 重新建立索引

    def sigmoid(self, inx):
        """
        the activation function
        :param inx: the independent variable inx
        :return:
        """
        return 1.0 / (1.0 + math.exp(-inx))

    def init_weight(self):
        """
        init weights and threshold of the neural in the range 0-1.
        :return: None
        """
        weights_input_hidden = np.random.sample((self.options['input_nodes'], self.options['hidden_nodes']))
        weights_hidden_output = np.random.sample((self.options['hidden_nodes'], self.options['output_nodes']))

        threshold_hidden = np.random.random(self.options['hidden_nodes'])
        threshold_output = np.random.random(self.options['output_nodes'])

        self.weights['input_hidden'],self.weights['hidden_output'] = weights_input_hidden, weights_hidden_output
        self.threshold['hidden'], self.threshold['output'] = threshold_hidden, threshold_output

    def calculate_output(self, x):
        """
        calculate the output of hidden layer and output layer
        :param x: the input value, its dimension is the input_nodes*1.
        :return: None
        """
        b = [];
        # 遍历隐藏层结点，得到每一个隐藏层的输出
        for h in range(self.options['hidden_nodes']):
            temp = 0;
            # 针对该隐藏层结点，计算所有输入节点加权和
            for i in range(self.options['input_nodes']):
                # 输入层第i个神经元与隐藏层第h个神经元之间加权和，得到第h个神经元接收到的输入
                temp += self.weights['input_hidden'][i][h] * x[i]

            # b[h]为第h个神经元的输出，使用sigmoid函数,求接收到的输入与该神经元阈值的差值的sigmoid函数。
            b.append(self.sigmoid(temp - self.threshold['hidden'][h]))

        y = [];
        # 遍历输出结点
        for o in range(self.options['output_nodes']):
            temp = 0;
            for h in range(self.options['hidden_nodes']):
                # 使用隐藏层神经元h与输出层神经元o之间的权重与隐藏层h的输出b[h]
                temp += self.weights['hidden_output'][h][o] * b[h];

            y.append(self.sigmoid(temp - self.threshold['output'][o]))

        self.output['b'] = b
        self.output['y'] = y

    def get_gradient(self, x, label):
        """
        get the gradient of the output neuron and hidden neuron
        :param label:
        :return:
        """
        # 遍历输出结点，针对每一个预测输出进行计算tmp，也就是书本上的公式5.10
        y = self.output['y']
        g = []
        for j in range(self.options['output_nodes']):
            # 最后一个因子是本来的输出与神经网络输出的差值
            temp = y[j] * (1 - y[j]) * (label - y[j])
            g.append(temp)

        #     print(g)

        # 书本上公式5.15
        b = self.output['b']
        e = []
        for h in range(self.options['hidden_nodes']):
            tmp = 0
            for j in range(self.options['output_nodes']):
                # 隐藏层与输出层之间的权重与g的乘积进行累加。
                temp += self.weights['hidden_output'][h][j] * g[j]
            e.append(b[h] * (1 - b[h]) * temp)

        self.gradient['output'], self.gradient['hidden'] = g, e

    def update_weight_and_threshold(self, x):
        """
        update the weights and threholds
        :return: None
        """
        # 更新隐藏层和输出层之间的权重
        for h in range(self.options['hidden_nodes']):
            for j in range(self.options['output_nodes']):
                self.weights['hidden_output'][h][j] += self.options['learning_rate'] * self.gradient['output'][j] * self.output['b'][h]

        # 更新输出层的阈值
        for j in range(self.options['output_nodes']):
            self.threshold['output'][j] += -self.options['learning_rate'] * self.gradient['output'][j]

        # 更新输入层与隐藏层之间的权重
        for i in range(self.options['input_nodes']):
            for h in range(self.options['hidden_nodes']):
                self.weights['input_hidden'][i][h] += self.options['learning_rate'] * self.gradient['hidden'][h] * x[i]

        # 更新隐藏层的阈值
        for h in range(self.options['hidden_nodes']):
            self.threshold['hidden'][h] += -self.options['learning_rate'] * self.gradient['hidden'][h]

    def print_weight_and_threshold(self):
        # print("weights between input and hidden is")
        # print(self.weights['input_hidden'])

        print('output of output layer')
        print(self.output['y'])

    def calculate_loss(self, labels):
        loss = 0
        for i in range(self.options['output_nodes']):
            loss += (self.output['y'][i]-labels[i])**2
        return loss

    def graph(self, train_loss, valid_loss):
        """
        graph the loss
        :param train_loss:
        :param valid_loss:
        :return: None
        """
        plt.xlabel('epoch')
        plt.ylabel('loss')
        x = range(3000)
        plt.scatter(x, train_loss, c='red')
        plt.scatter(x, valid_loss, c='green')
        plt.grid(True)
        plt.show()

    def train(self):

        # preprocess data and split dataset
        self.split_dataset()

        # initialize weights and threshold
        self.init_weight()

        # begin to cycle
        epoch = 0
        train_loss_list = []
        valid_loss_list = []
        while epoch < 3000:
            train_loss = 0
            for i in range(len(self.data['train'])):
                x = np.array(self.data['train'][i:i + 1])
                label = self.label['train'][i]

                # calculate output of output layer and hidden layer
                self.calculate_output(*x)

                # get gredient
                self.get_gradient(*x, label)

                # update weights and threshold
                self.update_weight_and_threshold(*x)

                # print(label)
                loss = self.calculate_loss([label])
                train_loss += loss
            train_loss /= len(self.data['train'])
            train_loss_list.append(train_loss)
            # print("train_loss: %.5f" % train_loss)


            valid_loss = 0;
            for i in range(len(self.data['valid'])):
                x = np.array(self.data['valid'][i:i + 1])
                # print(self.label['valid'])
                label = self.label['valid'][i]

                # calculate output of output layer and hidden layer
                self.calculate_output(*x)

                # get gredient
                self.get_gradient(*x, label)

                # update weights and threshold
                self.update_weight_and_threshold(*x)

                # print(label)
                loss = self.calculate_loss([label])
                valid_loss += loss
            valid_loss /= len(self.data['train'])
            valid_loss_list.append(valid_loss)
            # print("valid_loss: %.5f" % valid_loss)

            # self.print_weight_and_threshold()
            epoch += 1
            if epoch %100== 0:
                print("traing %d times" % epoch)
        self.graph(train_loss_list, valid_loss_list)


if __name__ == '__main__':
    # create an BP object
    print("start training...")
    bp = BP(HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE, FILENAME)
    bp.train()

```

上面代码运行之后得到的图形如下图所示：
<img src="/images/DataMiningTheory/BP/BP_02.png" width="400px" height="400px" />

其中红色的先表示训练误差，绿色的线表示验证误差（这里没有说错）。至于为什么训练误差还会出现增大的情况，目前原因不是很清楚。思考可能的原因，如下：
- 数据集不够。因为使用的是书本上提供的数据集，共17条，训练集11条，验证集6条。
- 采用的标准BP算法，每一条数据都会对应的修改参数，从而可能存在抵消情况
