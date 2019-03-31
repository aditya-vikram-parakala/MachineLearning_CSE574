
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
tr_values=np.reshape(tr_values,(50000,784))
test_values=np.reshape(test_values,(10000,784))
#RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)
classifier2.fit(tr_values, tr_target)
p = classifier2.predict(test_values)
import pandas as pd
predicted_value = pd.DataFrame(p)
predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\randomforest_predvalues_mnist.csv")
from sklearn.metrics import accuracy_score
acc = accuracy_score(test_target, p)
print("accuracy MNIST dataset:",acc)
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(test_target, p)
print(cnf_matrix)
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
import pandas as pd
usps_predict = classifier2.predict(arr_mat)
# predicted_value = pd.DataFrame(usps_predict)
# predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\randomforest_predvalues_usps.csv")
arr_tar = np.asarray(USPSTar)
usps_predict.shape
acc = accuracy_score(arr_tar, usps_predict)
print("accuracy USPS dataset:",acc)
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(arr_tar, usps_predict)
print(cnf_matrix)
a = classifier2.score(arr_mat,arr_tar)
print(a)
#source: project3 methods pdf
