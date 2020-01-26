#!/usr/bin/env python
# coding: utf-8

# #CIS 419/519 HW0 iPython Notebook
#
# Complete the answers to Questions 5 and 6 by completing this notebook.
#

# # 5.) Dynamic Programming

# In[5]:



def CompletePath(s, w, h) -> str:
    '''This function is used to escape from a room whose size is w * h.
    You are trapped in the bottom-left corner and need to cross to the
    door in the upper-right corner to escape.
    @:param s: a string representing the partial path composed of {'U', 'D', 'L', 'R', '?'}
    @:param w: an integer representing the room width
    @:param h: an integer representing the room length
    @:return path: a string that represents the completed path, with all question marks in with the correct directions
    or None if there is no path possible.
    '''
    locationX = 0 #Track for location of X
    locationY = 0 #Track for location of Y
    history = [(0,0)] #List to store history location


    s,sL,sR,sU,sD = list(s),list(s),list(s),list(s),list(s)
    numberUnknown = s.count('?') #count number of numberUnknowns remain in the List

    ############################################################################################
    #The following section track the routing history, if it intersects itself, discard the data
    #When it reaches a new point, record the location in a list
    if numberUnknown == 0:

        for i in range(len(s)):
            if s[i] == 'L':
                locationX -=1
            elif s[i] =='R':
                locationX +=1
            elif s[i] == 'U':
                locationY +=1
            elif s[i] =='D':
                locationY -=1
            location = (locationX,locationY)
            if location in history:
                return(None)
            if locationX > w-1 or locationX < 0 or locationY > h-1 or locationY < 0:
                return(None)
            else:

                history.append(location)
    ############################################################################################
    ############################################################################################
    #The following section counts the number of steps to the right/top remain to get to destination
    if numberUnknown > 0:
        sL[s.index('?')] = 'L'
        sR[s.index('?')] = 'R'
        sU[s.index('?')] = 'U'
        sD[s.index('?')] = 'D'
    timesUp = s.count('U')
    timesDown = s.count('D')
    timesLeft = s.count('L')
    timesRight = s.count('R')
    up_Required = h - 1 + timesDown - timesUp
    right_Required = w - 1 + timesLeft - timesRight
    ############################################################################################
    ############################################################################################


    if numberUnknown == 0 and up_Required ==0 and right_Required==0: #If no unknown remains and alreay at destination, return the path
        return(''.join(s))
    elif numberUnknown ==0 and (up_Required!=0 or right_Required !=0): #if no unknown remains but not at destination, discard the path
        return(None)

    if s.index('?') == 0:
        return(CompletePath(''.join(sU), w, h) or CompletePath(''.join(sR), w, h)) #If the first unknown is at the beginning, the only possible path is to right or top
    elif s.index('?') == len(s)-1:
        return(CompletePath(''.join(sU), w, h) or CompletePath(''.join(sR), w, h)) #Same when the first unknown is at the end
    else:
        return(CompletePath(''.join(sU), w, h) or CompletePath(''.join(sR), w, h) or CompletePath(''.join(sD), w, h) or CompletePath(''.join(sL), w, h))
        #If the first unknown is in the middle of the path, check all four paths available recursively and pick one possible path


# # 6.) Pandas Data Manipulation

# In this section, we use the `Pandas` package to carry out 3 common data manipulation tasks :
#
# * **Calculate missing ratios of variables**
# * **Create numerical binary variables**
# * **Convert categorical variables using one-hot encoding**
#
# For the exercise, we will be using the Titanic dataset, the details of which can be found [here](https://www.kaggle.com/c/titanic/overview). For each of the data manipulation tasks, we have defined a skeleton for the python functions that carry out the given the manipulation. Using the function documentation, fill in the functions to implement the data manipulation.
#

# In[6]:


import pandas as pd
import numpy as np


# **Dataset Link** : https://github.com/rsk2327/CIS519/blob/master/train.csv
#
#
# The file can be downloaded by navigating to the above link, clicking on the 'Raw' option and then saving the file.
#
# Linux/Mac users can use the `wget` command to download the file directly. This can be done by running the following code in a Jupyter notebook cell
#
# ```
# !wget https://github.com/rsk2327/CIS519/blob/master/train.csv
# ```
#
#

# In[7]:


# Read in the datafile using Pandas
# data = pd.read_csv("train.csv")

# df = ...            # # TODO # #


# In[8]:


def getMissingRatio(inputDf):
    number_Missing = 0
    rows,columns = inputDf.shape #Get the number of rows and columns(features) of the table
    ratio_Array = []#The array which stores missing ratio for all the features
    for i in range(columns):
        for j in range(rows):
            if pd.isna(inputDf.at[j,inputDf.columns[i]]): #If the data at a certain location in a feature is missing , increase the missing number
                number_Missing +=1
        ratio_Array.append(number_Missing/rows) #Calculate the percentage missing based on the number missing
        number_Missing = 0 #Reset the number missing for another feature

    outDf = pd.DataFrame({'Feature':inputDf.columns,'MissingPercent':ratio_Array}) #Build a dataframe with missing features and missing rate

    return outDf



# In[9]:


def convertToBinary(inputDf, feature):
    rows,columns = inputDf.shape #
    for i in range(rows):
        if pd.isna(inputDf.loc[i,feature]):
            pass
        else :
            value_0 = inputDf.loc[i,feature]#Record the first value in the column
            break
    second_Value_Raised = False
    for i in range(rows):
        if inputDf.loc[i,feature] == value_0:
            inputDf.loc[i,feature] = 0
        elif inputDf.loc[i,feature] != value_0 and (not second_Value_Raised):
            value_1 = inputDf.loc[i,feature]#Record second value in the column
            second_Value_Raised = True
            inputDf.loc[i,feature] = 1
        elif inputDf.loc[i,feature] == value_1:
            inputDf.loc[i,feature] = 1
        elif pd.isna(inputDf.loc[i,feature]):
            pass
        else:
            print (f'{feature} is not a binary feature')# If there is a different value other than the first two, raise error
            return(0)

    outDf = inputDf
    return outDf



# In[10]:


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
    # if feature not in inputDf.columns:
    #     return('Feature not in dataset')
    # rows,columns = inputDf.shape
    # feature_List = []
    # OHE_Matrix = np.array([[]]) #Create a matrix to store the OHE values
    # for i in range(rows):
    #     if pd.isna(inputDf.loc[i,feature]):
    #         OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((1,len(feature_List)))),axis=0) #If missing data, create a new row of zeros
    #     elif str(inputDf.loc[i,feature]) not in feature_List:
    #         feature_List.append(str(inputDf.loc[i,feature]))
    #         OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((i+1,1))),axis=1)#if there is a new feature, create a new column of zeros
    #     if str(inputDf.loc[i,feature]) in feature_List:
    #         OHE_Matrix = np.concatenate((OHE_Matrix,np.zeros((1,len(feature_List)))),axis=0)#if this it is alreay in feature list , create a new row of zeros  and set the feature related column to 1
    #         OHE_Matrix[i,feature_List.index(str(inputDf.loc[i,feature]))]=1
    # for i in range(len(feature_List)):
    #     feature_List[i] = feature + '_'+feature_List[i]#New column names for OHE

    # OHE_Matrix = np.delete(OHE_Matrix,rows,0)#Delete the extra row created

    # dataOut= pd.DataFrame(OHE_Matrix,columns=feature_List) #Create a dataframe with OHE as matrix and the new feature list
    # outDf = pd.concat([inputDf,dataOut],axis=1)#Concate new features to original matrix
    # outDf = outDf.drop(feature,axis=1)#drop the original feature

    if feature not in inputDf.columns:
        raise ValueError('This is not in the feature list')
        return(None)

    return (pd.concat([inputDf,pd.get_dummies(inputDf.loc[:,feature],prefix=feature)],axis=1)).drop(feature,axis=1)


# In[11]:





# In[ ]:
