
import numpy as np
import pickle
import gzip
from sklearn.svm import SVC
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
#SVM
classifier1 = SVC(kernel='rbf',C=2, gamma = 0.05,verbose=1)
classifier1.fit(X, y)

p = classifier1.predict(test_values)

import pandas as pd
predicted_value = pd.DataFrame(p)
predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\svm_predvalues_mnist.csv")
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target, p)
print("accuracy MNIST dataset:",acc)

from sklearn.metrics import confusion_matrix
a = confusion_matrix(test_target,p)
print(a)

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
            
arr_mat = np.asarray(USPSMat)
print(arr_mat.shape)
USPS_X = np.reshape(arr_mat,(19999,784))
USPS_y = np.array(USPSTar)
p_usps = classifier1.predict(USPS_X)
acc_usps = accuracy_score(USPS_y, p_usps)
print("accuracy USPS dataset:",acc_usps)
# predicted_value = pd.DataFrame(p_usps)
# predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\svm_predvalues_usps.csv")
from sklearn.metrics import confusion_matrix
b = confusion_matrix(USPS_y,p_usps)
print(b)
#source: project3 methods pdf
