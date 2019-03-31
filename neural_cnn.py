#sourcs: refered stackoverflow and intro to data science page for few concepts nad a part of implementation

import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
import keras
#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)
img_rows = 28
img_cols = 28
if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = X_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#more reshaping
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

##model building
model =  tf.keras.Sequential()
#convolutional layer with rectified linear unit activation
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(tf.keras.layers.Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(tf.keras.layers.Flatten())
#fully connected to get all relevant data
model.add(tf.keras.layers.Dense(128, activation='relu'))
#one more dropout for convergence
model.add(tf.keras.layers.Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(tf.keras.layers.Dense(num_category, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

batch_size = 128
num_epoch = 10
#model training
model_log = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch
          ,verbose=1
          ,validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss MNIST:', score[0]) 
print('Test accuracy MNIST:', score[1])

import os
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig


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

target = np.asarray(USPSTar)
#target = np.reshape(target,(19999,10))
usps_target = keras.utils.to_categorical(target, num_category)
usps_target.shape
usps_data = np.asarray(USPSMat)
usps_data = usps_data.reshape(usps_data.shape[0], img_rows, img_cols,1)
usps_data.shape
#matrixvalues = np.reshape(arr_mat,(19999,784))
score_usps= model.evaluate(usps_data, usps_target, verbose=0)
print('Test accuracy USPSDATASET:', score_usps[1])

