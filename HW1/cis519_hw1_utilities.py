#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 Homework 1
# 
# Name: Yupeng Li
# 
# Pennkey: yupengli
# 
# PennID: 37169291

# In[ ]:


import random 
import numpy as np
import pandas as pd
from sklearn import tree
random.seed(42)  # don't change this line
def addDummyFeatures(inputDf, feature):
    """
    Create a one-hot-encoded version of a categorical feature and append it to the existing
    dataframe.

    After one-hot encoding the categorical feature, ensure that the original categorical feature is dropped
    from the dataframe so that only the one-hot-encoded features are retained.

    For more on one-hot encoding (OHE) : https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-yo u-have-to-use-it-e3c6186d008f

    Arguments:
        inputDf (Pandas.DataFrame): input dataframe
        feature (str) : Feature for which the OHE is to be performed


    Returns:
        outDf (Pandas.DataFrame): Resultant dataframe with the OHE features appended and the original feature removed

    """


    ## TODO ##
    if feature not in inputDf.columns:
        return('Feature not in dataset')
    rows,columns = inputDf.shape
    feature_List = []
    OHE_Matrix = np.array([[]]) #Create a matrix to store the OHE values
    for i in range(rows):
        if pd.isna(inputDf.loc[i,feature]):
            OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((1,len(feature_List)))),axis=0) #If missing data, create a new row of zeros
        elif str(inputDf.loc[i,feature]) not in feature_List:
            feature_List.append(str(inputDf.loc[i,feature]))
            OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((i+1,1))),axis=1)#if there is a new feature, create a new column of zeros
        if str(inputDf.loc[i,feature]) in feature_List:
            OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((1,len(feature_List)))),axis=0)#if this it is alreay in feature list , create a new row of zeros  and set the feature related column to 1
            OHE_Matrix[i,feature_List.index(str(inputDf.loc[i,feature]))]=1
    for i in range(len(feature_List)):
        feature_List[i] = feature + '_'+feature_List[i]#New column names for OHE

    OHE_Matrix = np.delete(OHE_Matrix,rows,0)#Delete the extra row created

    dataOut= pd.DataFrame(OHE_Matrix,columns=feature_List) #Create a dataframe with OHE as matrix and the new feature list
    outDf = pd.concat([inputDf,dataOut],axis=1)#Concate new features to original matrix
    outDf = outDf.drop(feature,axis=1)#drop the original feature
    return outDf



# ## **Preprocessing**
# 
# The first key step in any data modeling task is cleaning your dataset. Explore your dataset and figure out what sort of preprocessing is required. Good preprocessing can make or break your final model. So choose wisely.
# 
# Some of the preprocessing steps that you can consider are :
# 
# 
# *   One-hot encoding of variables
# *   Missing value imputation
# *   Removing outliers
# *   Converting binary features into 0-1 representation
# 
# 
# Feel free to reuse code you've already written in HW 0.
# 
# 
# 
# 
# 

    
# X = X.fillna(0)


# In[ ]:



# 

# In[ ]:


def cross_validated_accuracy(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
   random.seed(random_seed)
   y = pd.DataFrame(y)
   y = y.iloc[:,0]
   # corrects = 0
   # testedTotal = 0
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
           model = DecisionTreeClassifier.fit(sampleDf,sampley)
           predictedy = model.predict(testDf)
           # print(f'Trial {i} Fold {j}')
           # print((predictedy == testy).sum()/len(testy))
           accuracy_Arr[i,j] = (predictedy ==testy).sum()/len(testy)
           # corrects += (predictedy == testy).sum()
           # testedTotal += len(testy)
   cvScore = accuracy_Arr.sum()/(num_trials*num_folds)
   
   # cvScore = corrects/testedTotal
   x_bar = np.mean(accuracy_Arr)
   S = np.std(accuracy_Arr)
   n = len(accuracy_Arr)
   t = 2.626
   confidence = (x_bar+t*S/np.sqrt(n),x_bar-t*S/np.sqrt(n))
   
   print(cvScore)




   return(cvScore)
  # """
  #  Args:
  #       DecisionTreeClassifier: An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")
  #       X: Input features
  #       y: Labels
  #       num_trials: Number of trials to run of cross validation
  #       num_folds: Number of folds (the "k" in "k-folds")
  #        random_seed: Seed for uniform execution (Do not change this) 

  #   Returns:
  #       cvScore: The mean accuracy of the cross-validation experiment

  #   Notes:
  #       1. You may NOT use the cross-validation functions provided by Sklearn
  # """
    # Shuffle the dataframe




  ## TODO ##


  


# In[ ]:


def automatic_dt_pruning(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
  random.seed(random_seed)
  ccp_alpha_List = np.linspace(0,1,101)
  accuracyList = []
  for ccp_alpha in ccp_alpha_List:
      print(f'ccp_alpha = {ccp_alpha}')
      DecisionTreeClassifier.set_params(ccp_alpha = ccp_alpha)
      accuracy_Run = cross_validated_accuracy(DecisionTreeClassifier,X, y, num_trials, num_folds, random_seed)
      print(accuracy_Run)
      if len(accuracyList) > 0:
          if accuracyList[-1]-accuracy_Run > 0.005:
              return (ccp_alpha - 0.01)
      accuracyList.append(accuracy_Run)
      
  ccp_alpha = ccp_alpha_List[-1]
  
  """
  Returns the pruning parameter (i.e., ccp_alpha) with the highest cross-validated accuracy

  Args:
        DecisionTreeClassifier  : An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")      
        X (Pandas.DataFrame)    : Input Features
        y (Pandas.Series)       : Labels
        num_trials              : Number of trials to run of cross validation
        num_folds               : Number of folds for cross validation (The "k" in "k-folds") 
        random_seed             : Seed for uniform execution (Do not change this)


    Returns:
        ccp_alpha : Tuned pruning paramter with highest cross-validated accuracy

    Notes:
        1. Don't change any other Decision Tree Classifier parameters other than ccp_alpha
        2. Use the cross_validated_accuracy function you implemented to find the cross-validated accuracy

  """


  ## TODO ##



  return ccp_alpha


# ## **Tuning and Testing**
# 
# With the helper functions and your processed dataset, build a Decision Tree classifier to classify Diabetic patients and tune it to maximize model performance.
# 
# Once you are done with your modeling process, test your model on the test dataset and output your predictions in a file titled "cis519_hw1_predictions.csv", with one row per prediction.

# In[ ]:
