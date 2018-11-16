import numpy as np
from math import log
def loadDataSet():
    postingList=[['my','dog','flea','problem','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec
## 创建词库
def createVocablist(dataset):
    #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    vocablist=set([])
    for doc in dataset:
        vocablist=vocablist | set(doc)
        return list(vocablist)
## 创建词向量
def setofWord2Vec(vocabList,inputSet):
    #vocabList为词库，inputSet为输入的词列表
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in  vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in  my Vocabulary!" % word)
    return returnVec
##朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0_Vec=np.zeros(numWords)
    p1_Vec=np.zeros(numWords)
    p0_num=0
    p1_num=0
    for i in range(numTrainDocs):
        if trainMatrix[i]==1:
            p1_Vec+=trainMatrix[i]
            p1_num+=1
        else:
            p0_Vec+=trainMatrix[i]
            p0_num+=1
    p0_prob=p0_Vec/p0_num
    p1_prob=p1_Vec/p0_num
    return p0_prob,p1_prob,pAbusive
## 测试代码
def classifyNB(vec2Classify,p0_prob,p1_prob,pClass1):
    """

    :param vec2Classify: 要测试的向量
    :param p0_prob: 类0对应的类条件概率
    :param p1_prob: 类1对应的类条件概率
    :param pClass1: 类1对应的先验概率
    :return: 分类结果
    """
    p1=sum(vec2Classify*p1_prob)+log(pClass1)
    p0=sum(vec2Classify*p0_prob)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0



