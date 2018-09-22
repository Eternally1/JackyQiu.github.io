---
title: java基础系列知识——collections
date: 2018-08-10 16:54:29
categories: 
- java
tags: ["java","collections"]
---

# java基础系列知识——collections
主要是自己在写java代码过程中使用到的一些java集合的相关内容。

# 1 Map
## 1.1 TreeMap
TreeMap的默认排序规则：按照key的字典顺序来进行升序。  
下面的代码是，通过输入n对数字，每一对数字是key value组成，将key值一样的value进行求和，然后根据key值排序
```java
import java.util.*;
int n = scan.nextInt();
Map<Integer,Integer> map = new TreeMap<Integer,Integer>();
for(int i = 0;i<n;i++){
    int key = scan.nextInt();
    int value = scan.nextInt();
    if(map.containsKey(key)){
        map.put(key, value+map.get(key));
    }else{
        map.put(key, value);
    }
}
for(Integer i:map.keySet()){
    System.out.println(i+" "+map.get(i));
}
```

## 1.2 HashMap
HashMap 是一个散列表，它存储的内容是键值对(key-value)映射。 在HashMap中通过get()来获取value，通过put()来插入value，ContainsKey()用来检验对象是否已经存在。LinkedHashMap保存了记录的插入顺序，在使用Iterator遍历的时候，保证了元素迭代的顺序。而HashMap遍历的时候获取的数据完全是随机的。
```java
HashMap<String,String> hm = new LinkedHashMap<String,String>();
```

# 2 Set
## 2.1 TreeSet
TreeSet可用于对象元素的排序的，同时保证元素唯一。  
下面的代码是随机输入n个数字，然后对着n个数字进行去重，然后排序。
```java
int n = scan.nextInt();
int[] temp = new int[n];
for(int i = 0;i<n;i++){
    temp[i] = scan.nextInt();
}
//开始去重和排序,使用TreeSet
TreeSet<Integer> ts = new TreeSet<Integer>();
for(Integer j:temp){
    ts.add(j);
}
Iterator iter = ts.iterator();   //数据遍历
while(iter.hasNext()){			
    System.out.println(iter.next());
}
```
将上面的ts转换为字符串数组
```java
String[] res = (String[])ts.toArray(new String[ts.size()])
```



