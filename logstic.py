import numpy as np
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    # labelMat 为列向量
    labelMat=np.mat(classLabels).transpose()
    # m,n 分别为样本数量和单一样本的特征数量
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycle=50
    weights=np.zeros((n,1))
    for k in range(maxCycle):
        y_hat=sigmoid(dataMatrix*(weights))
        error=labelMat-y_hat
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

