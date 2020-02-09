#!/usr/bin/env python
# coding: utf-8

# # CIS 519 HW 2

# In[ ]:


import pandas as pd
import random
import numpy as np
from numpy import linalg as LA
from numpy.linalg import *

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# # Linear Regression

# In[ ]:


'''
    Linear Regression via Gradient Descent
'''

class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
        self.JHist = []
        for i in range(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta.T: ", theta.T)
            yhat = X*theta
            theta = theta -  (X.T * (yhat - y)) * (self.alpha / n)
        return theta
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** Not returning a matrix with just one value! **
        '''
        n,d = X.shape
        yhat = X*theta
        J =  (yhat-y).T * (yhat-y) / n
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas Dataframe
            y is an n-dimensional Pandas Series
        '''
        n = len(y)
        X = X.to_numpy()
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term

        y = y.to_numpy()
        n,d = X.shape
        y = y.reshape(n,1)

        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))

        self.theta = self.gradientDescent(X,y,self.theta)   


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas DataFrame
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        X = X.to_numpy()
        X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
        return pd.DataFrame(X*self.theta)


# ### Test code for linear regression

# In[ ]:


def test_linreg(n_iter = 2000):
  # load the data
  filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-multivariateData.csv"
  df = pd.read_csv(filepath, header=None)

  X = df[df.columns[:-1]]
  y = df[df.columns[-1]]

  n,d = X.shape

  # # Standardize features
  from sklearn.preprocessing import StandardScaler
  standardizer = StandardScaler()
  X = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization

  # # initialize the model
  init_theta = np.matrix(np.random.randn((d+1))).T
  alpha = 0.01

  # # Train the model
  lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
  lr_model.fit(X,y)

  # # Compute the closed form solution
  X = np.asmatrix(X.to_numpy())
  X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
  y = np.asmatrix(y.to_numpy())
  n,d = X.shape
  y = y.reshape(n,1)
  thetaClosedForm = inv(X.T*X)*X.T*y
  print("thetaClosedForm: ", thetaClosedForm.T)


# # Run the Linear Regression Test

# In[ ]:


# test_linreg(2000)


# # Polynomial Regression

# In[ ]:


'''
    Template for polynomial regression
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8, tuneLambda = True, regLambdaValues=[]):
        '''
        Constructor
        '''
        self.alpha = 0.4
        self.theta = np.zeros(degree+1)
        self.regLambda = regLambda
        self.degree = degree
        self.n_iter = 2000
        self.regLambdaValues = regLambdaValues
        #TODO


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d data frame, with each column comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 data frame
            degree is a positive integer
            
        '''
        poly_X_arr = np.zeros((X.shape[0],degree))
        for i in range(degree):
            poly_X_arr[:,i] = X.iloc[:,0]**(i+1)
            
        poly_X =pd.DataFrame(poly_X_arr, columns =[i+1 for i in range(degree)])
        
            
            
        
        
        return poly_X
            
        #TODO
        

    def fit(self, X, y):
        '''
    
            Trains the model
            Arguments:
                X is a n-by-1 data frame
                y is an n-by-1 data frame
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling first
        '''
        #TODO
        X = self.polyfeatures(X,self.degree)
        for i in range(X.shape[1]):
            mu_J = X.iloc[:,i].mean()
            s = X.iloc[:,i].std()
            X.iloc[:,i]= X.iloc[:,i].apply(lambda x : (x-mu_J)/s)
        X = X.to_numpy()
        X = np.c_[np.ones((X.shape[0],1)), X]     # Add a row of ones for the bias term

        y = y.to_numpy().flatten()
        
        self.theta = self.gradientDescent(X,y,self.theta)   

            

    
    def gradientDescent(self, X, y, theta):
        self.JHist = []
        # y = y.reshape(len(y),1)
        for i in range(self.n_iter):
            # self.JHist.append( (self.cost(X, y, theta), theta) )
            # print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta.T: ", theta.T)


            # theta[0] = theta[0] -self.alpha/X.shape[0]*(np.matmul(X,theta)-y.reshape(y.shape[0])).sum()
            # theta[1:len(theta)] = theta[1:len(theta)]*(1-self.alpha*self.regLambda)-self.alpha/X.shape[0]*np.sum((hypo-y)*X,axis=0)[1:len(theta)]
            for j in range(X.shape[1]):
                hypo = np.matmul(X,theta)
                cost = self.cost(X, y, theta)
                print(cost)
                if j ==0:
                    theta[j]=theta[j]-self.alpha/X.shape[0]*((hypo-y).sum())
                    pass
                else:
                    theta[j] = theta[j]*(1-self.alpha*self.regLambda) - self.alpha/X.shape[0]*((hypo-y)*X[:,j]).sum()
                    # print(theta)
        # theta = np.concatenate(([self.theta_0],theta),axis=None)
 
        

        return theta
    
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        # TODO
        X = self.polyfeatures(X,self.degree)
        for i in range(X.shape[1]):
            mu_J = X.iloc[:,i].mean()
            s = X.iloc[:,i].std()
            X.iloc[:,i]= X.iloc[:,i].apply(lambda x : (x-mu_J)/s)
        X = X.to_numpy()

        X = np.c_[np.ones((X.shape[0],1)), X]     # Add a row of ones for the bias term
        return pd.DataFrame(np.matmul(X,self.theta))

    def cost(self,X,y,theta):
        
        # X = self.polyfeatures(X, self.degree)
        # for i in range(X.shape[1]):
        #     mu_J = X.iloc[:,i].mean()
        #     s = X.iloc[:,i].std()
        #     X.iloc[:,i]= X.iloc[:,i].apply(lambda x : (x-mu_J)/s)
        # X = X.to_numpy()
        # X = np.c_[np.ones((X.shape[0],1)), X]     # Add a row of ones for the bias term

        # y = y.to_numpy().flatten()
        
        hypo = np.matmul(X,theta)
        
        cost = 1/X.shape[0]*((hypo-y)**2).sum()+self.regLambda * (theta[1:]**2).sum()
        return cost


    def autoTuning(self, X,y):
        return None
        
    def cross_validated_accuracy(self, X, y, num_trials, num_folds, random_seed, regLambda):
       random.seed(random_seed)
       accuracy_Arr = np.zeros((num_trials,num_folds))
       for i in range(num_trials):
           index = [q for q in range(X.shape[0])]
           random.shuffle(index)
           shuffledDf = X.set_index([index]).sort_index()
           sudoY = y.copy()
           sudoY.index = index
           shuffledy = sudoY.sort_index()
           dfList = np.array_split(shuffledDf,num_folds)
           yList = np.array_split(shuffledy,num_folds)  
           for j in range(num_folds):
               dfs = dfList.copy()
               ys = yList.copy()
               testDf = dfs.pop(j)
               testy = ys.pop(j)
               sampleDf = pd.concat(dfs,axis=0)
               sampley = pd.concat(ys,axis=0)
               self.fit(sampleDf,sampley)
               accuracy_Arr[i,j] = self.cost(testDf,testy,self.theta)
       cvScore = accuracy_Arr.sum()/(num_trials*num_folds)
       
       print(cvScore)
      
       return(cvScore)

        


# # Test Polynomial Regression

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def test_polyreg_univariate():
    '''
        Test polynomial regression
    '''

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-polydata.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree = d, regLambda = 0.001)
    # model.cross_validated_accuracy(X, y, 10, 10, 42)
    model.fit(X, y)
    
    # output predictions
    xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, 'rx')
    plt.title('PolyRegression with d = '+str(d))
    plt.plot(xpoints, ypoints, 'b-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


# In[ ]:


test_polyreg_univariate()

