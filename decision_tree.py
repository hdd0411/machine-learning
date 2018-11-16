from math import log
import operator
def cal_shannon_entropy(dataset):
    """
    :param dataset: list,every line stands for a sample
    :return: entropy
    """
    num_samples=len(dataset)
    ## 字典，键表示类别，键值表示相应类别的个数
    label_counts={}
    for sample in dataset:
        ## 每个样本记录得最后一列表示类别
        current_label=sample[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label]=0
        label_counts[current_label]+=1
    shannon_entropy=0
    for key in label_counts:
        prob=float(label_counts[key])/num_samples
        shannon_entropy-=prob*log(prob,2)
    return shannon_entropy

def splitdataset(dataset,axis,value):
    """
    :param dataset:  list,every line stands for a sample
    :param axis:  which colomn to split
    :param value: the splitted union have the same property as value
    :return: dataset
    """
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis]==value:
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataset):
    """
    :param dataset: list,every line stands for a sample
    :return: best property to split dataset
    """
    numfeatures=len(dataset[1])-1
    baseEntropy=cal_shannon_entropy(dataset)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numfeatures):
        featlist=[example[i] for example in dataset] ## axis=i:属性的所有值
        uniqueVals=set(featlist) ## unique values in  featlist
        newEntropy=0
        for value in uniqueVals:
            sub_dataset=splitdataset(dataset,i,value)  ## 依次按轴axis，和属性值value划分数据集
            prob=len(sub_dataset)/float(len(dataset))  ## 子数据集个数占总样本数据个数的比率
            newEntropy+=prob*cal_shannon_entropy(sub_dataset)
        infoGain=baseEntropy-newEntropy  ## 计算信息增益
        ## 寻找使得信息增益最大的属性值
        if(infoGain>bestInfoGain):
            bestInfoGain = infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
        ##字典按照键值排序，降序排列，返回的是列表，列表中每一个元素都是元组
        sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]
def createTree(dataset,labels):
     #labels:是指dayaset的属性列表
     ##决策树停止划分的两个条件
     #condition1:
     #集合中的所有元素同属一类
     classList=[example[-1] for example in dataset]
     if classList.count(classList[0]) ==len(classList):
         return classList[0]
     #condition2:
     #集合中所有特征都被使用完，无法继续划分
     if len(dataset[0])==1:
         return majorityCnt(classList)
     bestFeat=chooseBestFeatureToSplit(dataset)
     bestFeatLabel=labels[bestFeat]
     myTree={bestFeatLabel:{}}
     ### 以上代码为了创建决策树的根节点
     del(labels[bestFeat])
     featValues=[example[bestFeat] for example in dataset]
     uniqueVals=set(featValues)
     for value in uniqueVals:
         subLabels=labels[:]
         myTree[bestFeatLabel][value]=createTree(splitdataset(dataset,bestFeat,value),subLabels)
     return myTree



### 模型保存及加载
def storeTree(inputTree,filename):
    import pickle
    fw=open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr=open(filename)
    return pickle.load(fr)











