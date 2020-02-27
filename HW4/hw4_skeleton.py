#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 
# #**Homework 4 : Adaboost and the Challenge**

# In[ ]:


import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler


# # Adaboost-SAMME

# In[ ]:


import numpy as np
import math
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor

        Class Fields 
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''

        self.clfs = [None] * numBoostingIters  # keep the class fields, and be sure to keep them updated during boosting
        self.betas = []
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        
        #TODO



    def fit(self, X, y, random_state=None):
        '''
        Trains the model. 
        Be sure to initialize all individual Decision trees with the provided random_state value if provided.
        
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        #TODO
        X = X.to_numpy()
        backup_X = X.copy()
        y = pd.DataFrame(y)
        y = y.to_numpy().flatten()
        np.place(y,y==0,-1)
        K = len(set(y))
        weights = np.full(X.shape[0],1/X.shape[0])
        for t in range(self.numBoostingIters):
            self.clfs[t] = tree.DecisionTreeClassifier(criterion="entropy", max_depth = self.maxTreeDepth)
            self.clfs[t].fit(X,y,sample_weight = weights)
            predicted_y = self.clfs[t].predict(X)
            weighted_Error = ((predicted_y != y)*weights).sum()
            beta = 1/2*(np.log((1-weighted_Error)/weighted_Error)+np.log(K-1))
            self.betas.append(beta)
            weights = weights * np.exp(-beta*y*predicted_y)
            weights = 1/weights.sum() * weights 

        return None
        
            # weights = (weights - weights.mean)/weights.std()

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
            
        '''
        #TODO
        X = X.to_numpy()
        prediction = np.zeros(X.shape[0])
        for t in range(self.numBoostingIters):
            prediction += self.betas[t]*self.clfs[t].predict(X)
        prediction =  np.sign(prediction)
        np.place(prediction,prediction==-1,0)
        return prediction
            
            


# # Test BoostedDT

# In[ ]:


import numpy as np
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_boostedDT():

  # load the data set
  sklearn_dataset = datasets.load_breast_cancer()
  # convert to pandas df
  df = pd.DataFrame(sklearn_dataset.data,columns=sklearn_dataset.feature_names)
  df['CLASS'] = pd.Series(sklearn_dataset.target)
  df.head()

  # split randomly into training/testing
  train, test = train_test_split(df, test_size=0.5, random_state=42)
  # Split into X,y matrices
  X_train = train.drop(['CLASS'], axis=1)
  y_train = train['CLASS']
  X_test = test.drop(['CLASS'], axis=1)
  y_test = test['CLASS']


  # train the decision tree
  modelDT = DecisionTreeClassifier()
  modelDT.fit(X_train, y_train)

  # train the boosted DT
  modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
  modelBoostedDT.fit(X_train, y_train)

  # train sklearn's implementation of Adaboost
  modelSKBoostedDT = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=100)
  modelSKBoostedDT.fit(X_train, y_train)

  # output predictions on the test data
  ypred_DT = modelDT.predict(X_test)
  ypred_BoostedDT = modelBoostedDT.predict(X_test)
  ypred_SKBoostedDT = modelSKBoostedDT.predict(X_test)

  # compute the training accuracy of the model
  accuracy_DT = accuracy_score(y_test, ypred_DT)
  accuracy_BoostedDT = accuracy_score(y_test, ypred_BoostedDT)
  accuracy_SKBoostedDT = accuracy_score(y_test, ypred_SKBoostedDT)

  print("Decision Tree Accuracy = "+str(accuracy_DT))
  print("My Boosted Decision Tree Accuracy = "+str(accuracy_BoostedDT))
  print("Sklearn's Boosted Decision Tree Accuracy = "+str(accuracy_SKBoostedDT))
  print()
  print("Note that due to randomization, your boostedDT might not always have the ")
  print("exact same accuracy as Sklearn's boostedDT.  But, on repeated runs, they ")
  print("should be roughly equivalent and should usually exceed the standard DT.")

test_boostedDT()

