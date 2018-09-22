---
title: java基础知识系列——demo
date: 2018-08-10 16:42:50
categories:
- java
tags: ["java","realize"]
---

# 引言
本文主要是关于java算法练习过程中，遇到的一些不错的解决思路，整理下来。

# 方法
## 求质数的因子
```java
//参数x是要求的数，返回的res是因子，中间以空格分隔开
public static String getResult(long x){
    String res = "";
    for(int i = 2;i<x;i++){
        if(x%i == 0){
            //此时说明该数是它的一个因子
            x = x/i;
            res += i+" ";
            i = 1;   //i要重置为2
        }
        
    }
    //最好找不到除数的时候，剩下的数就是质因子了。
    res += x;
    return res;
}
```

## 将数逆序输出并使它不重复
通过将数逆序输出，同时重复的不输出。如果是400200，那么输出的是24，当然如果想输出024，可以判断一下原始数据的最后一位数，然后决定输出结果。
```java
//比如输入 786884  则输出 4867
//经典之处在于，通过一个数组来记录该数字是否已经出现过，从而筛选输出的结果。
public static int getUnrepeatedInteger(int x){
    int[] a = new int[10]; 
    for(int i = 0;i<10;i++){
        a[i] = 0;
    }
    int num = 0;
    while(x!=0){
        if(a[x%10] == 0){
            //判断x的最低位是否已经出现过
            a[x%10]++;
            num = num*10+(x%10);	
        }
        x = x/10;
    }
    return num
```

## 判断字符串中是否含有长度超过2的相同子串
```java
//判断是否包含相同子串
for(int i = 0;i<s.length()-2;i++){
	String substring = s.substring(i,i+3);
	if(s.substring(i+1).contains(substring)){
		//判断该子串之后的字符串中是否包含相同的子串即可
		return false;
	}
}
return true;

```

