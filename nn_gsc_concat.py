
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#read the data
data_x= np.genfromtxt(r'C:\Users\aditya vikram\GSC_X_gsc_concat.csv', delimiter=',')
data_y= np.genfromtxt(r'C:\Users\aditya vikram\GSC_t_gsc_concat.csv',delimiter=',')
# Get the data values
X = data_x[:, 0:1024]
# Get the target
Y = data_y[:]



#Split the data in train & test
Y_reshape = Y.reshape(Y.shape[0], 1)
x_train, x_test, y_train, y_test = train_test_split(X, Y_reshape)

print ("x_train shape: " + str(x_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("x_test shape: " + str(x_test.shape))
print ("y_test shape: " + str(y_test.shape))
# 1024 feartures for gsc concat will be read and passed to our model
num_features = x_train.shape[1]


# In[9]:


learning_rate = 0.05
training_epochs = 100 

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, num_features], name="X")
# we have binary classification same writer or diff writer 0 or 1
Y = tf.placeholder(tf.float32, [None, 1], name="Y")

# Initialize our weigts & bias
#bias of 1 is added 
W = tf.get_variable("W", [num_features, 1], initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", [1], initializer = tf.zeros_initializer())

Z = tf.add(tf.matmul(X, W), b)
prediction = tf.nn.sigmoid(Z) # sigmoid is the activation function used

# Calculate the cost
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Z, labels = Y))

# Use Adam as optimization method
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# initialize all the global variables.
init = tf.global_variables_initializer()

cost_history = np.empty(shape=[1],dtype=float)
# run the session
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={X: x_train, Y: y_train})
        cost_history = np.append(cost_history, c)
        
        
    # Calculate the correct predictions
    correct_prediction = tf.to_float(tf.greater(prediction, 0.5))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.to_float(tf.equal(Y, correct_prediction)))
    # print out the accuracy
    print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
    print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
    #source: referred to github repo for the implementation

