---
title: java基础系列知识——Snippets
date: 2018-08-10 16:41:29
categories: 
- java
tags: ["java","小demo"]
---

# java基础系列知识——Snippets
主要是java编程中遇到的一些小的问题收集，便于之后进行查阅

## 1 小demo
### 1.1 split
```java
// 注意： . 、 | 和 * 等转义字符，必须得加 \\。
// 注意：多个分隔符，可以用 | 作为连字符。
String[] s = temp.split("\\."); //在以“.”作为切分的符号时，注意转义
String[] sr = s.split("\\s+");    //匹配多个空格的时候
String temp = "a and b or c";
String[] s = temp.split("and|or")
//输出结果为  a  b  c
String s = "E:\\abc\\def\\fgj\\re.txt";
String[] s = s.split("\\\\");   //注意文件路径时如何进行分割
```

### 1.2 进制转换
```java
// 十六进制、八进制、二进制转换成十进制，使用Integer.parseInt(number, base)
int x = Integer.parseInt(s,2)    //此时说明s是二进制的格式，转换成10进制，存储在x中

//十进制转换成十六进制、八进制、二进制
Integer s = (Integer)scan.nextInt();    //s需要是Integer类型，而不能是int
String a = s.toHexString(s);
String b = s.toBinaryString(s);   //转化成2进制
String c = s.toOctalString(s);   //转换成8进制。

```

### 1.3 获取字符串中的第一个字符
```java
char c = s.charAt(0)   //使用charAt()方法获取指定索引位置的字符
```

### 1.4 字符的ASCII编码
```java
char c = 'b';
int a = (int)c;
```

### 1.5 字符数组转换成字符串
```java
// c是字符数组
String s = new String(c);
```

### 1.6 将数组按照字典进行排序，直接使用sort()
```java
//s是要进行排序的数组，不用返回值。
Arrays.sort(s); 
```

### 1.7 数值类型和字符串相互转换
```java
int num = Integer.parseInt("123")
double doubleNum = Double.parseDouble("1.23")

// 数值类型转换成string
String s = Integer.toString(123)
```

### 1.8 substring
public String substring(int beginIndex[, int endIndex])返回字符串的子字符串，beginIndex是起始索引（包括），endIndex是结束索引（不包括）
```java
String s = "WWW.baidu.com";
s.substring(3);   //baidu.com
s.substring(3,9);  //baidu
```

### 1.9 数组初始化
静态初始化和动态初始化：
- 一维数组
- 二维数组

```java
//一维数组  静态初始化
int[] arr = {1,2,3};

//一维数组  动态初始化
int[] arr = new int[3];
arr[0] = 1; 

//二维数组静态初始化
int[][] arr = {{1,2,3},{4,5,6}}
````


