
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


logreg = pd.read_csv(r"C:\Users\aditya vikram\Desktop\logreg_predvalues_mnist.csv", usecols=['0'])
nn = pd.read_csv(r"C:\Users\aditya vikram\Desktop\nn_predvalues_mnist.csv", usecols=['0'])
randforest = pd.read_csv(r"C:\Users\aditya vikram\Desktop\randomforest_predvalues_mnist.csv",usecols=['0'])
svm = pd.read_csv(r"C:\Users\aditya vikram\Desktop\svm_predvalues_mnist.csv",usecols=['0'])


# In[3]:


alogreg = logreg.values.T
ann = nn.values.T
rf = randforest.values.T
sv = svm.values.T


# In[4]:


con = np.concatenate((alogreg,ann,rf,sv),axis=0)


# In[28]:


import numpy as np
import pickle
import gzip
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
y = test_target


# In[29]:


from scipy import stats
m = stats.mode(con)


# In[30]:


pred_val_mvotig = m[0]


# In[31]:


#pred_val_mvotig.shape[1]


# In[34]:


cnt=0
for i in range(pred_val_mvotig.shape[1]):
    if pred_val_mvotig[0][i] == y[i]:
        cnt+=1        
#print(cnt)
print("accuracy MNIST",cnt/pred_val_mvotig.shape[1])


# In[49]:


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
usps_y = np.asarray(USPSTar)


# In[50]:


logreg_usps = pd.read_csv(r"C:\Users\aditya vikram\Desktop\logreg_predvalues_usps.csv", usecols=['0'])
nn_usps = pd.read_csv(r"C:\Users\aditya vikram\Desktop\nn_predvalues_usps.csv", usecols=['0'])
randforest_usps = pd.read_csv(r"C:\Users\aditya vikram\Desktop\randomforest_predvalues_usps.csv",usecols=['0'])
svm_usps = pd.read_csv(r"C:\Users\aditya vikram\Desktop\svm_predvalues_usps.csv",usecols=['0'])


# In[53]:


u_alogreg = logreg_usps.values.T
u_ann = nn_usps.values.T
u_rf = randforest_usps.values.T
u_sv = svm_usps.values.T


# In[54]:


u_con = np.concatenate((u_alogreg,u_ann,u_rf,u_sv),axis=0)
#u_con


# In[55]:


from scipy import stats
u_m = stats.mode(u_con)
pred_val_mvotig = u_m[0]


# In[57]:


c=0
for i in range(pred_val_mvotig.shape[1]):
    if pred_val_mvotig[0][i] == usps_y[i]:
        c+=1        
print(c)
print("accuracy USPS",c/pred_val_mvotig.shape[1])

