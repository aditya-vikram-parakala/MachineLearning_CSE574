
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
# imported required packages
# few declarations for further calculations.
maxAcc = 0.0
maxIter = 0
C_Lambda = 0.02
TrainingPercent = 80 # 80% of raw data 
ValidationPercent = 10 # 10% of raw data
TestPercent = 10  #10% of raw data 
M = 2  # It is the complexity of the model,we can change this acccordingly by checking the accuracy of the model to improve it later on.
PHI = []
IsSynthetic = False
def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:  
            t.append(int(row[0]))
    #print("Raw Training Generated..")
    return t  # we will get the values of the target variable i.e. 0,1,2 

def GenerateRawData(filePath, IsSynthetic):    
    dataMatrix = [] 
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        for row in reader:
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)   
    
    #if IsSynthetic == False : #this is for deleting the columns in our data that contains 0's which would not contribute to calculation of the varience and is not invertable.
        #dataMatrix = np.delete(dataMatrix, [5,6,7,8,9], axis=1)# we deletd 5 cols so left with 41 features out of 46 features.
    dataMatrix = np.transpose(dataMatrix)  #we transpose the data matrix to simplify the further steps of matrix multiplication   
    #print ("Data Matrix Generated..")
    return dataMatrix # each data row we have 1x41

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): #given to use 80% of the dataset as training
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01))) #calculate the length of target training set
    t           = rawTraining[:TrainingLen] # loading the elements till the training length it has only one column
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t 

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

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic): # generating covarience matrix
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct)) # calculate the varience of features wrt each other 
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]+0.02 # fill the diagonal elements with the varience values and rest all are 0's and add a bias term of 0.2 to remove singular value erroe
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma) # for calculation multiplied by a scalar
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)   # calculate (x-mu)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T) # after all matrix multiplications it produces a scalar value
    return L # we get a scalar value at last i.e. 1x1 matrix

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv)) # gaussian radial basis function
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma) # moore penrose pseudo inverse calculation
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI #we get a design matrix

def GetWeightsClosedForm(PHI, T, Lambda): # to get weights using closed form solution
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) # output of vaidation tesing set
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct): # calculate the accuracy of the model i.e root mean squared error
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) 
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) # produces the Erms 
# locate and prepare the dataset for starting the process of learning
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

# calculation of the weights using closed form approach 
ErmsArr = []
AccuracyArr = []
# clustering the training data to find out the means of the clusters through the centroids of the partitions. 
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_
# generating all the values required using the moore penrose pseudo inverse   
BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)

# Graddient descent approach for calculating the optimal weights 
#W = np.zeros(shape=(10,1024))
W_Now        = np.dot(220, W)
La           = 2
learningRate = 0.01 # this factor is used to update the wieghts in every other step of gradient descent 
L_Erms_Val   = [] # empty lists to append data when calculated in further steps
L_Erms_TR    = []
L_Erms_Test  = []
W_Mat        = []

# iterationfor the datapoints
for i in range(0,400):
    
    
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    
    #-----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
    Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))
    
    #-----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
    Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))
    
    #-----------------TestingData Accuracy---------------------#
    TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
    Erms_Test = GetErms(TEST_OUT,TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))

print ('----------Gradient Descent Solution--------------------')
#print ("M = 10 \nLambda  = 0.0001\neta=0.05") # the model complexity is M 
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


# In[3]:


plt.plot(L_Erms_Test,"r-")
plt.xlabel("no of iterations")
plt.ylabel("error")
plt.title("Erms-Test")
plt.show()
plt.plot(L_Erms_Val)
plt.xlabel("no of iterations")
plt.ylabel("error")
plt.title("Erms-Validation")
plt.show()
plt.plot(L_Erms_TR)
plt.xlabel("no of iterations")
plt.ylabel("error")
plt.title("Erms-Train")
plt.show()

