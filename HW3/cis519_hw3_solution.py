#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 
# #**Homework 3 : Logistic Regression**

# In[ ]:


import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler


# ### Logistic Regression

# In[ ]:


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.00000001, regNorm=2, epsilon=0.0001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.regLambda = regLambda
        self.alpha = alpha 
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.initTheta = initTheta
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        y = y.reshape(-1)
        hypo = self.sigmoid(np.matmul(X,theta))
        
        if self.regNorm == 2:
            cost = -(y*np.log(hypo) + (1-y)*np.log(1-hypo)).sum() + regLambda*np.linalg.norm(theta[1:])**2  
        elif self.regNorm ==1:
            cost = -(y*np.log(hypo) + (1-y)*np.log(1-hypo)).sum() + regLambda * abs(theta[1:]).sum()
        
        return cost 
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        
        '''
        
        # y = y.flatten()
        y = y.reshape(-1)
        
        hypo = self.sigmoid(X@theta).reshape(len(y),1)
        if self.regNorm ==2:
            regulated =  ((hypo-y.reshape(len(y),1))*X).sum(axis=0) + regLambda*self.theta
            regulated[0] = regulated[0] - regLambda * theta[0]
            return regulated
        elif self.regNorm == 1:
            regulated =  ((hypo-y.reshape(len(y),1))*X).sum(axis=0) + regLambda*np.sign(self.theta)
            regulated[0] = regulated[0] - regLambda*np.sign(theta[0])
            return regulated
         
            

                
                


    def fit(self, X, y):
        
        X = X.to_numpy()
        X = np.c_[np.ones((X.shape[0],1)), X] #Add a column of one as bias
        y = y.to_numpy().flatten()
        
        if self.initTheta is None:
            self.initTheta = np.zeros(X.shape[1])
            
        self.theta = self.initTheta.copy()
        
        last_theta = np.zeros(X.shape[1])

        for i in range(self.maxNumIters):
            self.theta = self.theta - self.alpha *self.computeGradient(self.theta, X, y, self.regLambda)
            # print(theta)
            # print(last_theta)
            if np.linalg.norm(self.theta - last_theta) < self.epsilon:
                break
            else:
                last_theta = self.theta.copy()
        
        return None

        
        
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        # for i in range(X.shape[0]):
        #     X.iloc[:,1] = X.iloc[:,i].apply(lambda x : (x-self.mu_J[i])/self.s[i])
        X = X.to_numpy()

        X = np.c_[np.ones((X.shape[0],1)), X]
        
        predictions = self.sigmoid(np.matmul(X,self.theta))
        for i in range(predictions.shape[0]):
            if predictions[i] >= 0.5:
                predictions[i] = 1
            elif predictions[i]< 0.5:
                predictions[i]  = 0
        return pd.DataFrame(predictions)
            

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''

        # for i in range(X.shape[0]):
        #     X.iloc[:,1] = X.iloc[:,i].apply(lambda x : (x-self.mu_J[i])/self.s[i])
        X = X.to_numpy()

        X = np.c_[np.ones((X.shape[0],1)), X]
        return pd.DataFrame(np.matmul(X,self.theta))



    def sigmoid(self, Z):

        return 1/(1+np.exp(-Z))


# # Test Logistic Regression 1

# In[ ]:


# Test script for training a logistic regressiom model
#
# This code should run successfully without changes if your implementation is correct
#
from numpy import loadtxt, ones, zeros, where
import numpy as np
from pylab import plot,legend,show,where,scatter,xlabel, ylabel,linspace,contour,title
import matplotlib.pyplot as plt

def test_logreg1():
    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data1.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape

    # # Standardize features
    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    Xstandardized = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    logregModel = LogisticRegression(regLambda = 50,regNorm = 2)
    logregModel.fit(Xstandardized,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()


# 

# # Map Feature

# In[ ]:


def mapFeature(X, column1, column2, maxPower = 6):
    '''
    Maps the two specified input features to quadratic features. Does not standardize any features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the maxPower polynomial
        
    Arguments:
        X is an n-by-d Pandas data frame, where d > 2
        column1 is the string specifying the column name corresponding to feature X1
        column2 is the string specifying the column name corresponding to feature X2
    Returns:
        an n-by-d2 Pandas data frame, where each row represents the original features augmented with the new features of the corresponding instance
    '''
    total_Degrees = np.array([i for i in range(maxPower+2)]).sum()-1
    map_Array = np.zeros((X.shape[0],total_Degrees))
    counter =0
    for i in range(1,maxPower+1):
        for j in range(0,i+1):
            map_Array[:,counter] = X.iloc[:,1]**(j) * X.iloc[:,0]**(i-j)
            counter +=1


            
    mapFeature = pd.DataFrame(map_Array)

    return(mapFeature)
            


# # Test Logistic Regression 2

# In[ ]:


from numpy import loadtxt, ones, zeros, where
import numpy as np
from pylab import plot,legend,show,where,scatter,xlabel, ylabel,linspace,contour,title
import matplotlib.pyplot as plt

def test_logreg2():

    polyPower = 6

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data2.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape

    # map features into a higher dimensional feature space
    Xaug = mapFeature(X.copy(), X.columns[0], X.columns[1], polyPower)

    # # Standardize features
    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    logregModel = LogisticRegressionAdagrad(regLambda = 1E-9, regNorm=2)
    logregModel.fit(Xaug,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = mapFeature(allPoints, allPoints.columns[0], allPoints.columns[1], polyPower)
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # standardize data
    
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.show()


    print(str(Z.min()) + " " + str(Z.max()))




# # Logistic Regression with Adagrad

# In[ ]:


class LogisticRegressionAdagrad:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.00001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.regLambda = regLambda
        self.alpha = alpha 
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.initTheta = initTheta
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        y = y.reshape(-1)
        hypo = self.sigmoid(np.matmul(X,theta))
        
        if self.regNorm == 2:
            cost = -(y*np.log(hypo) + (1-y)*np.log(1-hypo)).sum() + regLambda*np.linalg.norm(theta[1:])**2  
        elif self.regNorm ==1:
            cost = -(y*np.log(hypo) + (1-y)*np.log(1-hypo)).sum() + regLambda * abs(theta[1:]).sum()
        
        return cost 
    
    
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        # y = y.flatten()
        
        # last_theta = np.zeros(X.shape[1])
        # G = np.zeros(X.shape[1])
        # alphas = np.zeros(X.shape[1])

        # for runs in range(self.maxNumIters):
        #     combined =  np.concatenate((X,y.reshape(len(y),1)),axis=1)
        #     np.random.shuffle(combined)
        #     X,y = combined[:,:-1],combined[:,-1]
        #     # self.alpha = 0.1/(runs + 10)
        #     for i in range(20):

        #         hypo = self.sigmoid(np.inner(X[i,:],theta))
        #         for j in range(X.shape[1]):
        #             # print(G[j])
        #             G[j]+=(( hypo-y[i])*X[i,j])**2
        #             alphas[j] = self.alpha/np.sqrt(G[j])
        #             if j== 0:
        #                 theta[j] = theta[j]-alphas[j]*(hypo-y[i])

        #             else:
        #                 theta[j] = theta[j]*(1-alphas[j]*self.regLambda)-alphas[j]*(hypo-y[i])*X[i,j]
        #     # print(np.linalg.norm(theta-last_theta))
        #     if np.linalg.norm(theta - last_theta) < self.epsilon:
        #         break
        #     else:
        #         last_theta = theta.copy()

            
        # return theta
        y = y.reshape(-1)
    
        hypo = self.sigmoid(X@theta)
        if self.regNorm ==2:
            regulated =  ((hypo-y)*X) + regLambda*self.theta
            regulated[0] = regulated[0] - regLambda * theta[0]
            return regulated
        elif self.regNorm == 1:
            regulated =  (hypo-y)*X + regLambda*np.sign(self.theta)
            regulated[0] = regulated[0] - regLambda*np.sign(theta[0])
            return regulated
         
                
        
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        X = X.to_numpy()
        X = np.c_[np.ones((X.shape[0],1)), X] #Add a column of one as bias
        y = y.to_numpy().flatten()
        
        combined =  np.concatenate((X,y.reshape(len(y),1)),axis=1)
        np.random.shuffle(combined)
        X,y = combined[:,:-1],combined[:,-1]

        
        # last_theta = np.zeros(X.shape[1])
        # G = np.zeros(X.shape[1])
        # alphas = np.zeros(X.shape[1])
        
        if self.initTheta is None:
            self.initTheta = np.zeros(X.shape[1])
            
        self.theta  = self.initTheta.copy()
            
            
        for runs in range(1000):
            for i in range(X.shape[0]):
                self.theta = self.theta - self.alpha*self.computeGradient(self.theta, X[i,:], y[i], self.regLambda)
                
        
  


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        X = X.to_numpy()

        X = np.c_[np.ones((X.shape[0],1)), X]
        
        predictions = self.sigmoid(np.matmul(X,self.theta))
        for i in range(predictions.shape[0]):
            if predictions[i] >= 0.5:
                predictions[i] = 1
            elif predictions[i]< 0.5:
                predictions[i]  = 0
        return pd.DataFrame(predictions)

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''

        X = X.to_numpy()

        X = np.c_[np.ones((X.shape[0],1)), X]
        return pd.DataFrame(self.sigmoid(np.matmul(X,self.theta)))

    def sigmoid(self, Z):

        return 1/(1+np.exp(-Z))

# test_logreg1()
# test_logreg2()


def learningCurve(RegressionMethod):
    return None


def comparingRegression():
    df = pd.read_csv('hw3-diabetes.csv',header = None)
    print('Percentage of instances with missing features:')
    print(df.isnull().sum(axis=0)/df.shape[0])
    print()
    print('Class information:')
    print(df.iloc[:,df.shape[1]-1].value_counts())
    