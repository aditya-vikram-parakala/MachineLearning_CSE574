
# coding: utf-8

# In[1]:


import numpy as np
import csv 
import random
import math
import pandas as pd


# In[2]:


TrainingPercent = 80 # 80% of raw data 
ValidationPercent = 10 # 10% of raw data
TestPercent = 10  #10% of raw data 
IsSynthetic =False
def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    #changedif IsSynthetic == False : #this is for deleting the columns in our data that contains 0's which would not contribute to calculation of the varience and is not invertable.
        changeddataMatrix = np.delete(dataMatrix, [0,10], axis=1)# we deletd 5 cols so left with 41 features out of 46 features.
    dataMatrix = np.transpose(changeddataMatrix)  #we transpose the data matrix to simplify the further steps of matrix multiplication   
    #print ("Data Matrix Generated..")
    return dataMatrix # each data row we have 1x41
#print(Data_values.shape)
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len] # generating the training data matrix
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): #
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t
#Data_1= GenerateRawData(r'C:\Users\aditya vikram\humandata_X_hd_concat.csv',IsSynthetic=False)
#X = GenerateTrainingDataMatrix(Data_1, TrainingPercent )

def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t  # we will get the values 
#target_values =GetTargetVector(r'C:\Users\aditya vikram\humandata_t_hd_concat.csv')
#y = GenerateValTargetVector(target_values, ValPercent, TrainingCount)

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #given to use 80% of the dataset as training
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #calculate the length of target training set
    t           = rawTraining[:TrainingLen] # loading the elements till the training length it has only one column
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t 


def GenerateValData(rawData, ValPercent, TrainingCount): #
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


# In[3]:


#TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
#TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)


# In[4]:


RawTarget = GetTargetVector(r'C:\Users\aditya vikram\GSC_t_gsc_concat.csv')
RawData   = GenerateRawData(r'C:\Users\aditya vikram\GSC_X_gsc_concat.csv',IsSynthetic)
#RawData = RawData.loc[:, (~RawData.isin([0])).any(axis=0)]
#RawData[~np.all(r == 0, axis=1)]
# preparing the data of taining i.e. training data , training target accordingly
TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)
# preparing the validation data 
ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)
#Preparing the test data 
TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)

X=np.transpose(TrainingData)
X_val=np.transpose(ValData)
X_test=np.transpose(TestData)
y=TrainingTarget
y_val=ValDataAct
y_test =TestDataAct
print(y.shape)
print(y_val.shape)
print(y_test.shape)


# In[ ]:


#source intro to data science website, referenced a part of the code 


# In[5]:


class LogisticRegression:
   def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
       self.lr = lr
       self.num_iter = num_iter
       self.fit_intercept = fit_intercept
   
   def __add_intercept(self, X):
       intercept = np.ones((X.shape[0], 1))
       return np.concatenate((intercept, X), axis=1)
   
   def __sigmoid(self, z):
       return 1 / (1 + np.exp(-z))
   def __loss(self, h, y):
       return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
   
   def fit(self, X, y):
       if self.fit_intercept:
           X = self.__add_intercept(X)
       
       # weights initialization
       self.theta = np.zeros(X.shape[1])
       
       for i in range(self.num_iter):
           z = np.dot(X, self.theta)
           h = self.__sigmoid(z)
           gradient = np.dot(X.T, (h - y)) / y.size
           self.theta -= self.lr * gradient
           
           #if(self.verbose == True and i % 10000 == 0):
               #z = np.dot(X, self.theta)
               #h = self.__sigmoid(z)
               #print(f'loss: {self.__loss(h, y)} \t')
   
   def predict_prob(self, X):
       if self.fit_intercept:
           X = self.__add_intercept(X)
   
       return self.__sigmoid(np.dot(X, self.theta))
   
   def predict(self, X, threshold):
       return self.predict_prob(X) >= threshold


# In[16]:


model = LogisticRegression(lr=0.1, num_iter=3000)
get_ipython().run_line_magic('time', 'model.fit(X, y)')
preds = model.predict(X, 0.5)
# accuracy
(preds == y).mean()


# In[17]:


model = LogisticRegression(lr=0.1, num_iter=3000)
get_ipython().run_line_magic('time', 'model.fit(X_val, y_val)')
preds = model.predict(X_val, 0.5)
# accuracy
(preds == y_val).mean()


# In[18]:


model = LogisticRegression(lr=0.1, num_iter=3000)
get_ipython().run_line_magic('time', 'model.fit(X_test, y_test)')
preds = model.predict(X_test, 0.5)
# accuracy
(preds == y_test).mean()

