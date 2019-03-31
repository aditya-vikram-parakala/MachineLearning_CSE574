import tensorflow as tf
import os
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import random
from sklearn.metrics import accuracy_score
import pandas as pd 

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0       #normalization of pixels 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),    #changed to relu,sigmoid,tanh
  tf.keras.layers.Dropout(0.2),                         #to prevent overfitting
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)   #output layer
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
log = model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
# plotting the metrics of the model
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(log.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.subplot(2,1,2)
plt.plot(log.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.tight_layout()
fig

preds_1 = model.predict(x_test)  #predict the output values
new_1 = np.zeros((len(preds_1)))
for i in range(len(preds_1)):
    new_1[i]=np.argmax(preds_1[i])  # take the element with max probability value in each row and store it in a new array
#predicted_value = pd.DataFrame(new_1)
#predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\nn_predvalues_mnist.csv")
acc_1 = accuracy_score(y_test,new_1)
print("accuracy MNIST dataset:",acc_1)
from sklearn.metrics import confusion_matrix
b = confusion_matrix(y_test,new_1)
print(b)

from PIL import Image
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
#print(arr_mat.shape)
matrixvalues = np.reshape(arr_mat,(19999,28,28))
#matrixvalues.shape

score = model.evaluate(matrixvalues, USPSTar)

preds = model.predict(matrixvalues)

new = np.zeros((len(preds)))
for i in range(len(preds)):
    new[i]=np.argmax(preds[i])    
#predicted_value = pd.DataFrame(new)
#predicted_value.to_csv(r"C:\Users\aditya vikram\Desktop\nn_predvalues_usps.csv")


usps_tar = np.asarray(USPSTar)


acc = accuracy_score(usps_tar, new)
print("accuracy USPS dataset:",acc) #accuracy usps data
from sklearn.metrics import confusion_matrix
a = confusion_matrix(usps_tar,new)
print(a)

