---
title: c语言中关于串的相关知识以及操作
date: 2018-07-19 19:43:02
categories: 
- C语言
tags: ["C语言","字符串"]
comments: true  # 可评论
toc: true # 显示文章目录
---

### 1 串的基本概念  
    
 串，即是字符串，由零个或者多个字符组成的有限序列，是数据元素为单个字符的特殊线性表。一般记为：S1='a1a2a3a4a5....an'。
  
### 2 串的存储结构： 
       
   定长顺序存储结构、堆分配存储结构和块链存储结构三种。
        
#### a.*定长顺序存储结构*
定长顺序存储结构是用一组地址连续的存储单元存储串值的字符序列，就是将串定义成字符串数组。数组的名字就是串名。数组的上界预先给出，所以也称为静态存储。

存储结构定义如下：       
```C
#define MAXL 256
typedef unsigned char SString[MAXL+1];//0号单元用于存储串长，串值从1号单元开始放。
另一种是从0号单元开始存储串值。结构定义如下：
#define MAXL 60
typedef struct{
    char str[MAXL];
    int length;
}SString;
```
此种存储结构有来两个缺点：
一是需要预先定义一个串允许的最大长度，当MAXL估计过大的时候串的存储密度就会降低，会浪费较多空间；二是由于限定了串的最大长度，使串的某些运算，比如联接收到一定限制。

#### b.*堆分配存储结构存储*
它其实也是利用一组地址连续的存储单元存储串值的字符序列，但是存储空间是在程序运行的时候动态分配的。因此可以利用c语言中动态分配函数库中的malloc()来分配空间，还可以利用realloc()增加空间
 
存储结构定义如下：
```C
typedef struct
{
    char *ch;
    int length;
}HString;
```
#### c.*块链存储结构*
是使用链式存数结构存储串，每个节点有data域和next指针域组成。
     
```C
#define CHUNKSIZE 80
typedef struct Chunk
{
    char ch[CHUNKSIZE];
    struct CHunk *next;
}Chunk;
typedef struct
{
    Chunk *head,*tail;
    int curlen;
}LString;
```

### 串的相关操作：
      
#### 1.串赋值算法： 
	    

```C
#include<stdio.h>
#define MAXL 256
typedef unsigned char str[MAXL];

void strAssign(str &T, char *chars)
{
    int i = 0;
    T[0] = 0;//0号单元存储字串长度
    for ( i = 0; chars[i]; i++)
    {
        T[i + 1] = chars[i];//用字符数组chars给串赋值.
    }
    T[0] = i;
}

void main()
{
    str T;
    char chars[] = "abcdefghijk";
    strAssign(T, chars);
    printf("串长是%d\n ", T[0]);
    printf("赋值后的串是 ：");
    for (int i = 1; i <= T[0]; i++)
    {
        printf("%c", T[i]);
    }
}
```

#### 2.求子串算法：
```C
#include<stdio.h>
#define MAXL 256
#define ERROR 0
#define OK 1

typedef int Status;
typedef unsigned char str[MAXL];

void strAssign(str &T, char *s)
//用字符数组给T赋值
{
    int i = 0;
    T[0] = 0;
    for (; s[i]; i++) T[i + 1] = s[i];
    T[0] = i;

}

Status subString(str &sub, str T, int pos, int len)
//用sub返回第 pos个字符起长度为len的子串。
{
    if (pos<1 || pos>T[0] || len<0 || len>T[0] - pos + 1)
        return ERROR;
    for (int i = 1; i <= len; i++)
    {
        sub[i] = T[pos + i - 1];
    }
    sub[0] = len;
}

void main()
{
    int pos, len;
    str T, sub;
    char chars[100];
    printf("请输入字符串 ");
    //scanf_s("%[^\n]", chars, sizeof(chars));//[^\n]只有遇到回车才会停止读入.
    gets_s(chars);//此处若是使用scanf_s()如上面注释的那样，也是没问题的.
    strAssign(T, chars);
    printf("请输入子串开始的位置和长度（中间用逗号隔开） ");
    scanf_s("%d,%d", &pos, &len);
    getchar();
    if (subString(sub, T, pos, len))//判断是否取子串成功.
    {
        printf("从第%d位置开始，长度为%d的子串为 ", pos, len);
        for (int i = 1; i <= sub[0]; i++)
        {
            printf("%c", sub[i]);
        }
        printf("\n");
    }
    else
        printf("求子串失败.....\n");

}
```

#### 3.串比较算法:
  

```C
#include<stdio.h>
#include<stdlib.h>

#define OK 1
#define ERROR 0
#define OVERFLOW -1

typedef int Status;
//串的堆分配存储表示
typedef struct
{
	char *ch;
	int length;
}HString;

Status strAssign(HString &s, char *chars)
{
    int i;
    char *c = chars;
    for (i = 0; *c; i++, c++);//求chars的长度
    if (!i)
    {
        s.ch = NULL;
        s.length = 0;
    }
    else
    {
        s.ch = (char*)malloc(i*sizeof(char));
        if (!(s.ch))exit(OVERFLOW);
        for (int j = 0; j < i; j++)
        {
            s.ch[j] = chars[j];
        }
        s.length = i;
    }
    return OK;
}

Status strCompareTo(HString a, HString b)
//若a<b,则则返回值<0,若a>b，则返回值>0,若a=b,则返回值=0；
{
    int i = 0;
    for (int i = 0; i < a.length&&b.length; i++)
    {
        if (a.ch[i] != b.ch[i])
            return (a.ch[i] - b.ch[i]);
    }
    return (a.length - b.length);
}

void main()
{
    HString Sa, Sb;
    char char_a[100], char_b[100];
    printf("请输入字符串a:");
    gets_s(char_a);
    strAssign(Sa, char_a);
    printf("请输入字符串b:");
    gets_s(char_b);
    strAssign(Sb, char_b);
    if (strCompareTo(Sa, Sb) == 0)
    printf("串 %s 等于串 %s", char_a, char_b);//此处是char_a,char_b,不能是Sa，Sb.
    if (strCompareTo(Sa, Sb) < 0)
    printf("串 %s 小于串 %s", char_a,char_b);
    if (strCompareTo(Sa, Sb)>0)
    printf("串 %s 等于串 %s", char_a, char_b);

}
```
#### 4.串联接算法

```C
#include <stdio.h>

#define TRUE 1
#define FALSE 0
#define MAXL 255

typedef int Status;
typedef unsigned char SString[MAXL + 1];

void strAssign(SString &T, char *s)
//用字符数组s给串T赋值
{
    T[0] = 0;
    int i = 0;
    for (; s[i]; i++)
    {
        T[i + 1] = s[i];
    }
    T[0] = i;
}

//定义了一个int型变量uncut，用于判断是否会被截断。
//uncun = TRUE时未被截断，uncut=FALSE时被截断。
Status connect(SString &T, SString S1, SString S2)
{
    int uncut, i;
    if (S1[0] + S2[0] <= MAXL)//未截断，注意0号单元存储的是串长.
    {
    for (i = 1; i <= S1[0]; i++)
        T[i] = S1[i];
    for (i = 1; i < S2[0]; i++)
        T[S1[0] + i] = S2[i];
    T[0] = S1[0] + S2[0];
    uncut = TRUE;
    }

    else if (S1[0] < MAXL)//S2被截断
    {
    for (i = 1; i <= S1[0]; i++)
        T[i] = S1[i];
    for (i = 1; i <= MAXL - S1[0]; i++)//注意此时的循环条件.
        T[S1[0] + i] = S2[i];
    T[0] = MAXL;
    uncut = FALSE;
    }

    else//S1,S2均被截断。
    {
    for (i = 1; i <= MAXL; i++)
        T[i] = S1[i];
    uncut = FALSE;
    }
    return uncut;
}

void main()
{
    SString S1, S2, T;
    char char_s1[100], char_s2[100];
    printf("请输入字符串S1:");
    gets_s(char_s1);
    printf("请输入字符串S2:");
    gets_s(char_s2);

    strAssign(S1, char_s1);
    strAssign(S2, char_s2);

    if (connect(T, S1, S2))
    {
        printf("S1= %s 和 S2= %s 联接过程中未被截断，连接后的串是: ", char_s1, char_s2);
        for (int i = 1; i <= T[0]; i++)
            printf("%c", T[i]);
    }
    else
    {
        printf("S1= %s 和 S2= %s 联接过程中被截断，连接后的串是: ", char_s1, char_s2);
        for (int i = 1; i <= T[0]; i++)
            printf("%c", T[i]);
    }
}
```

#### 5.串的模式匹配算法
  

```C
#include<stdio.h>

#define MAXL 255
#define OK 1
#define OVERFLOW -1


typedef unsigned char SString[MAXL+1];

void strAssign(SString &T, char *s)
//用字符数组s给串T赋值.
{
    int i = 0;
    T[0] = 0;//0号单元存储串长.
    for (; s[i]; i++)
    {
        T[i + 1] = s[i];
    }
    T[0] = i;
}

int index(SString T, SString S, int pos)
//返回子串S在主串T第pos个字符开始匹配的位置，若不存在，则返回0;
{
    int i = pos,j=1;
    while (i <= T[0] && j <= S[0])
    {
        if (T[i] == S[j])
        {
            i++;
            j++;
        }
        else
        {
            i = i - j + 2;
            j = 1;
        }
    }
    if (j > S[0])
        return i - S[0];
    else
        return 0;
}

void main()
{
    int pos;
    SString T, S;
    char char_a[100], char_b[100];
    printf("请输入主串A：");
    gets_s(char_a);
    printf("%s\n", char_a);
    printf("请输入主串B：");
    gets_s(char_b);
    printf("%s\n", char_b);

    strAssign(T, char_a);
    strAssign(S, char_b);

    printf("赋值成功！\n");

    pos = index(T, S, 1);
    if (pos)
    {
        printf("主串 T=%s 的子串 S=%s 在第%d个位置开始匹配。",char_a,char_b,pos);
    }
    else
        printf("主串 T=%s 和子串 S=%s 不匹配",char_a,char_b);
}
```
   以上就是串相关知识以及一些简单操作，代码处理平台是vs2013。初来乍到，多多关照，有什么写的不对的，欢迎指正。
