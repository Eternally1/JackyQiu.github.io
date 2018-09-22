---
title: python snippets
date: 2018-08-18 15:03:50
categories:
- Python
tags: ["Python","snippets"]
---

# 1汉字编码 URL编码解码


```python
s = u'长春疫苗'
s_code = s.encode('UTF-8')
s_code

# 如何对url链接中的汉字编码
import urllib
s_code1 = urllib.parse.quote(s)
print(s_code1)
s_code1 = "%25E9%25B9%25BF%25E6%2599%2597"
s_code2 = urllib.parse.unquote(s_code1)
# 解码
print(urllib.parse.unquote(s_code2))
```

    %E9%95%BF%E6%98%A5%E7%96%AB%E8%8B%97
    鹿晗
    

# 2 datetime求时间差


```python
import datetime
# 使用scrapy爬取数据，最后显示的有开始的时间和结束的时间，但是时间格式如下所示，要求时间差可以使用下面的方式。
finish_time = datetime.datetime(2018, 7, 25, 7, 5, 29, 616530)
start_time = datetime.datetime(2018, 7, 25, 7, 4, 57, 580133)
(finish_time-start_time).seconds
```
    32



# 3 python字符串相关

主要是字符串相关的一些知识点
- 访问字符串的值，字符串拼接，字符串运算
- 字符串格式化相关

## 3.1 字符串拼接，运算，访问


```python
s1 = "hello world";  
print(s1[4])   # 字符串访问

s2 = s1[:6]+"Qiu";   # 字符串拼接
print(s2);

# 重复输出字符串
s3 = s2*3;
print(s3)

# 原始字符串：所有的字符串都是直接按照字面意思来使用，没有转移特殊或者不能打印的字符。
s4 = r'\n\t\r hello world'
print(s4)
```

    o
    hello Qiu
    hello Qiuhello Qiuhello Qiu
    \n\t\r hello world
    

## 3.2 字符串格式化

看一些字符串格式化的规则：

%[(name)][flags][width].[precision]typecode

(name)      可选，用于选择指定的key  
flags          可选，可供选择的值有:
- \+       右对齐；正数前加正好，负数前加负号；
- \-        左对齐；正数前无符号，负数前加负号；
- 空格    右对齐；正数前加空格，负数前加负号；  
- 0        右对齐；正数前无符号，负数前加负号；用0填充空白处  

width         可选，占有宽度  
.precision   可选，小数点后保留的位数  

### 3.2.1 ASCII  二进制 十进制 八进制 十六进制


```python
d = 97
c1 = "%c" % d
print(c1)

# 转换成二进制，之后可以使用replace替换掉0b
c2 = bin(d)
print(c2)

c3 = "%o" % d;
print(c3)

# 大小写x都行，十六进制
c4 = "%X" % d;
print(c4)

# 浮点数
c5 = "%f" % d;
print(c5)

# 科学计数法
c6 = "%e" % d;
print(c6)

# 根据值得大小决定使用%f 还是%e
c7 = "%g" % d;
print(c7)
```

    a
    0b1100001
    141
    61
    97.000000
    9.700000e+01
    97
    

### 3.2.2 给变量命名


```python
age = 18;
name = "Q"
greeting = "his name is %(name)s, he is %(age)d years old" % {'name':name, 'age':age}
print(greeting)

# 下面这种既包含命名变量又包含没命名变量就不会了
# greeting1 = "his name is %(name)s, he is %(age)d years old and he is %s " % ({'name':name, 'age':age}, 'yes')
```

    his name is Q, he is 18 years old
    

### 3.2.3 填充0  左右对齐


```python
c = 1234;
# 用0填充空白处
c1 = "%010d" % c;
print(c1)


```

    0000001234
    

### 3.2.4 精度


```python
c = 1.234567

# 保留4位小数
c1 = "%.4f" % c;
print(c1)

# 保留四位小数，总长度为10,在左侧补0
c2 = "%010.4f" % c
print(c2)
```

    1.2346
    00001.2346 看空格
    

## 3.3 其他

将字符串转换成列表，元组


```python
s = "1234";
print(list(s), tuple(s))
```

    ['1', '2', '3', '4'] ('1', '2', '3', '4')
    

# 4 列表相关

## 4.1 获取列表中最大值所在的索引


```python
counts_baidu = [13879,17886,11305,5961,3086,2331,1680]
max_index = counts_baidu.index(max(counts_baidu))
max_index
```




    1



## 4.2 del  remove pop列表元素删除

- remove() 函数用于移除列表中某个值的第一个匹配项。该方法没有返回值，但是会移除列表中第一个匹配项
- pop() 函数用于移除列表中的一个元素（默认最后一个元素），并且返回该元素的值。



```python
l = [1,2,3,4,3,2,5]
l.remove(2)
l.pop(2)
del l[1]
```




    [1, 3, 2, 5]



# 5 range

## 5.1 生成数字列表

range(start, stop[, step])   注意返回的类型，要想看到输出的结果，可以使用list()，但是在使用的使用，可以不用list()进行转换


```python
c = range(10)
print(type(c))
print(list(c))

c1 = range(1,10)
print(list(c1))

c2 = range(1,11,2)
print(list(c2))
```

    <class 'range'>
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    [1, 3, 5, 7, 9]
    

# 6 zip

zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。


```python
a = [1,2,3]
b = [4,5,6]
zipped = list(zip(a,b))
print(zipped)

print(list(zip(*zipped)))
```

    [(1, 4), (2, 5), (3, 6)]
    [(1, 2, 3), (4, 5, 6)]
    

# 7 文件操作

## 7.1 按行读取文件内容 


```python
# delete stopwords
filepath = r"C:\others\doc\teamAndPersonInfo\研究生\资源库\stopwords\stopwords1.txt"
# read stopwords and create stopword set
def read_stopwords(filepath):
    sw_set = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        # traverse file by line
        for line in f:
            sw_set.add(line.strip("\n"))
    print("stopwords set:",sw_set)
    return sw_set
sw_set = read_stopwords(filepath)
```

    stopwords set: {'', "'", ':', '九', '嗳', '8', '5', ')', '6', '％', '哎呀', '"', '——', '月', '是', ' ]', '。', '啊', '!', '￥', '八', '>', '俺们', '）', '呗', '七', '］', '＞', '，', '唉', '还', '哎哟', '\u3000', '|', '７', '三', '〉', '９', '7', '＆', '１', '４', '/', '&', '3', '！', '２', '＊', '｛', '+', '\\', '9', ';', '：', '·', '［', '嗬', '＠', '\ufeff,', '*', '向', '｜', '’', '...', '‘', '2', '＜', '(', '的', '0', '咚', '—', '%', '喏', '＄', '很', '按', '`', '嗯', '在', '日', '>>', '六', '＃', '吧哒', '喔唷', '=', '啐', '；', '０', '？', '.', '３', '、', '五', '｝', '～', '4', '”', '》', '便', '二', '@', '$', '白', '_', '兮', '..', '哎', '８', '（', '?', '<', '了', '--', '给', '说', '零', '＋', '呃', '５', '着', '俺', '“', '#', '年', '…', '按照', '^', '和', '６', '阿', '1', ' [', 'A', '吧', '《', '〈', '到', '-', '︿', '咦', '尼', '~'}
    

# 8 外部包

## 8.1 collections.defaultdict

当键的默认值不清楚的时候，可以使用defaultdict，比如下面的这种情况。在key值不存在的时候，默认值为0。[参考这里](https://blog.csdn.net/the_little_fairy___/article/details/80551538)


```python
from collections import defaultdict
bag = ['apple', 'orange', 'cherry', 'apple','apple', 'cherry', 'blueberry']
count = defaultdict(lambda:0)    #   count = {};如果这么写，会报错keyerror
for fruit in bag:
    count[fruit] += 1
count
```




    defaultdict(<function __main__.<lambda>>,
                {'apple': 3, 'blueberry': 1, 'cherry': 2, 'orange': 1})



## 8.2 itertools排列组合

itertools.permutations(iterable, r=None) 其中r默认为空，当为空的时候，r的默认值是可迭代对象的长度。返回值是可迭代对象中元素的排列。
itertools.combinations(iterable, r)


```python
import itertools
l = ['A','B','C']
temp = itertools.permutations(l, 2)

print(list(temp))

temp = itertools.combinations(l,2)
print(list(temp))
```

    [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]
    [('A', 'B'), ('A', 'C'), ('B', 'C')]
    

# 9 dict

## 9.1 字典排序

sorted() 函数对所有可迭代的对象进行排序操作。
sorted(iterable[, key[, reverse]])
参数说明：

- iterable -- 可迭代对象。
- key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
- reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）

key是一个函数，用于指定进行比较的参数


```python
top_10_nouns = {'日报': 17, '疫苗': 1405, '记录': 143, '事': 42, '长生': 866, '狂犬': 89, '社会': 26, '声明': 14, '法律责任': 11, '教授': 6, '有效期': 11}

# 使用key进行排序
top_5 = sorted(top_10_nouns.items(), key=lambda x:x[1], reverse=True)[:5]
print(top_5)
print(top_5)
```

    [('疫苗', 1405), ('长生', 866), ('记录', 143), ('狂犬', 89), ('事', 42)]
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-c24851e2ff30> in <module>()
          6 
          7 # 使用cmp进行排序
    ----> 8 top_5 = sorted(top_10_nouns.items(), cmp=lambda x,y:cmp(x[1],y[1]))[:5]
    

    TypeError: 'cmp' is an invalid keyword argument for this function


# 10 lambda

如果需要，可以查看[这篇文档](http://python.jobbole.com/87848/)
