import numpy as np
import sys


class knn_clasifier:
    """knn classifier L2 distance
    input:
    train_x: (num,features)
    train_y: (num,) """
    def __init__(self,train_x,train_y):
        self.train_x=train_x
        self.train_y=train_y

    def predict(self,X,num_loop,k=1):
        """
        :param X: shape:(num,features)
        :param num_loop: train iter times
        :param k: select k nearest distance
        :return: predict
        """
        if num_loop==0:
            dist=self.computer_distance_no_loop(X)
        elif num_loop==1:
            dist=self.computer_distance_one_loop(X)
        elif num_loop==2:
            dist=self.computer_distance_two_loop(X)
        else:
            raise ValueError('Invalid value %d for loop'% num_loop)

    def computer_distance_no_loop(self,X):
        """

        :param X: test_samples,shape=(num,features)
        :return: distance:(num_test_samples,num_train_samples)
        """
        num_test=X.shape[0]
        num_train=self.train_x.shape[0]
        dist=np.zeros((num_test,num_train))
        T=np.sum(X**2,axis=1)
        F=np.sum(self.train_x**2,axis=1).T
        F=np.tile(F,(num_test,num_train))
        XF=2*X.dot(self.train_x.T)
        print(T.shape,F.shape,XF.shape)
        dist=T+F-XF
        return dist
    def computer_distance_one_loop(self,X):
        """

        :param X: test_samples,shape=(num,features)
        :return: distance:(num_test_samples,num_train_samples)
        """
        num_test=X.shape[0]
        num_train=self.train_x.shape[0]
        dist=np.zeros((num_test,num_train))
        for i in range(num_test):
            dist[i,:]=np.sum((X[i,:]-self.train_x)**2,axis=1).T
        return dist

    def computer_distance_two_loop(self,X):
        """

        :param X: test_samples,shape=(num,features)
        :return: distance:(num_test_samples,num_train_samples)
        """
        num_test=X.shape[0]
        num_train=self.train_x.shape[0]
        dist=np.zeros((num_test,num_train))
        for i in range(num_test):
            for j in range(num_train):
                dist[i,j]=np.sum((X[i,:]-self.train_x[j,:])**2)
        return dist

    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        Input:
        dists - A num_test x num_train array where dists[i, j] gives the distance
                between the ith test point and the jth training point.
        Output:
        y - A vector of length num_test where y[i] is the predicted label for the
            ith test point.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            ##np.argsort()升序排列，返回索引
            closest_y = self.train_y[np.argsort(dists[i, :])[:k]]
            ## np.unique() 返回u : 独一无二的数； 返回indices:旧列表元素在新列表中的位置，长度同closest_y
            u, indices = np.unique(closest_y, return_inverse=True)
            ##np.bincount 统计0-max(indices)出现的次数
            ##np.argmax返回最大值的索引
            y_pred[i] = u[np.argmax(np.bincount(indices))]















