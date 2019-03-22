from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###data prep###

#read csv data with pandas
data = pd.read_csv('Iris.csv')

#remove unnecessary columns
X = data.drop(labels=['Id', 'Species'], axis = 1).to_numpy()
#prepare classes (3 classes)
y = np.zeros((150, 3))
y[:,0] = (data.Species=='Iris-setosa').astype(int)
y[:,1] = (data.Species=='Iris-versicolor').astype(int)
y[:,2] = (data.Species=='Iris-virginica').astype(int)

#create test and training set
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]

#hyperparameters
learning_rate = 0.1
training_epochs = 1500
batch_size = 32

#define weights and bias unit and initialize them
W = tf.Variable(tf.zeros(shape=[4,3]))
b = tf.Variable(tf.zeros(shape=[1,3]))
x_batch = tf.placeholder(dtype=tf.float32, shape=[None, 4])
y_batch = tf.placeholder(dtype=tf.float32, shape=[None, 3])

#define model
model = tf.nn.softmax(tf.matmul(x_batch, W) + b)
#define cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=y_batch))
#define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize variables
init = tf.global_variables_initializer()

#Test model
correct = tf.equal(tf.argmax(model, 1), tf.argmax(y_batch, 1))
#accuracy
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#train the model
with tf.Session() as sess:
    sess.run(init)
    #mini batch gradient descent
    num_batches = math.ceil(len(train_X)/batch_size)
    print("Accuracy:", accuracy.eval({x_batch: test_X, y_batch: test_y}))
    for epoch in range(training_epochs):
        #shuffle trainings set (is this needed?)
        index = np.arange(len(train_X))
        np.random.shuffle(index)
        train_X = train_X[index]
        train_y = train_y[index]
        avg_cost = 0
        for i in range(num_batches):
            batch_xs = train_X[(i*batch_size):(i+1)*batch_size]
            batch_ys = train_y[(i*batch_size):(i+1)*batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={x_batch: batch_xs, y_batch:batch_ys})
            avg_cost += c / num_batches
        # output
        if (epoch + 1) % 300 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print("Accuracy:", accuracy.eval({x_batch: test_X, y_batch: test_y}))