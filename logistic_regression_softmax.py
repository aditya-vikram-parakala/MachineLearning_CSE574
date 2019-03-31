import pickle
import gzip
import numpy as np
import random
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
filename = r'C:\Users\aditya vikram\Desktop\ML_project3\mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
f.close()
#MNIST DATA PREPROCESSING
tr_values = np.asarray(training_data[0])
tr_target = np.asarray(training_data[1])
val_values = np.asarray(validation_data[0])
val_target = np.asarray(validation_data[1])
test_values =np.asarray(test_data[0])
test_target = np.asarray(test_data[1])
X=tr_values
y=tr_target
# adding the bias term 1 to both the training data and the testing data
X = np.insert(X, 0, 1, axis=1)
test_values = np.insert(test_values, 0, 1, axis=1)
#USPS DATA PREPROCESSING
from PIL import Image
import os
import numpy as np
USPSMat  = []
USPSTar  = []
curPath  = r'C:\Users\aditya vikram\Desktop\ML_project3\USPSdata\USPSdata\Numerals'
savedImg = []
for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)  
# conversting list to numpy array for easy calculation          
arr_mat = np.asarray(USPSMat)
USPS_X = np.reshape(arr_mat,(19999,784))
USPS_y = np.array(USPSTar)
USPS_X = np.insert(USPS_X, 0, 1, axis=1) # adding the bias term for testing data
#LOGISTIC REGRESSION 
#softmax function is the activation function we use for multi class classification problem
def smax(act):
    exp = np.exp(act)
    prob_val = np.zeros((act.shape))
    for i in range(act.shape[0]):
    #for j in range(act.shape[1]):
        prob_val[i,:]=exp[i,:]/np.sum(exp[i,:])
    return prob_val
# hot vector representation 
def one_of_k(y):
    result = np.zeros((y.shape[0],10))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if(j==(y[i])):
                result[i][j] = 1
    return result
# calculation of error after each iteration
def cal_error(pred_value,t_mat,X):
    t_mat = t_mat.reshape((50000,10))
    temp = np.matmul(X.T,pred_value-t_mat)
    return temp   
temp1 = []
# loss clac to know the convergence
def loss_calc(pred_value,t_mat):
    log_y = np.log(pred_value)
    loss_val = -(np.sum((t_mat*log_y)))/pred_value.shape[0]
    temp1.append(loss_val)
    return temp1   
def logistic_regression():
    num_iter = 500
    k=0
    lparam = 0.9
    LAMBDA= 0.001
    N = X.shape[0] # total number of samples
    wt = np.random.rand(785,10) # initialize random weights
    t_mat = one_of_k(y)
    lr = lparam/N # learning rate
    while(k<num_iter):
        act = np.matmul(X,wt)
        pred_value = smax(act)
        loss_val = loss_calc(pred_value,t_mat) 
        gradient = cal_error(pred_value,t_mat,X)
        reg_wt = LAMBDA * wt
        reg_wt[0,:] = 0
        wt =wt - lr *(gradient + reg_wt)
        k+=1
#     plt.plot(loss_val)
#     plt.xlabel('No of Iterations')
#     plt.ylabel('Loss')
#     plt.show()
    return wt # return the optimal weights after calculation
def accuracy_cal(X,y):
    wt_new = logistic_regression()
    final_val = smax(np.matmul(X,wt_new))
    pred_out = np.argmax(final_val,axis=1)
    #predicted_value = pd.DataFrame(pred_out)
    #predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\logreg_predvalues_usps.csv")
    from sklearn.metrics import confusion_matrix
    a = confusion_matrix(y,pred_out) # construct the confusion matrix
    print("confusion matrix",a)
    cnt=0
    for i in range(pred_out.shape[0]):
        if(pred_out[i]==y[i]):
            cnt+=1
    return cnt/(X.shape[0]) # calculating the accuracy
#MNIST dataset
acc_mnist= accuracy_cal(test_values,test_target)
print("ACCURACY MNIST: ",acc_mnist)
#USPS dataset
acc_usps= accuracy_cal(USPS_X,USPS_y)
print("ACCURACY USPS: ",acc_usps)

