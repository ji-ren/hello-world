#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

Weights_L1=tf.Variable(tf.random.normal([1,10]))
biases_L1=tf.Variable(tf.zeros([1,10]))
Neurons_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Neurons_L1)

Weights_L2=tf.Variable(tf.random.normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
Neurons_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Neurons_L2)

loss=tf.reduce_mean(tf.square(y-prediction))

init=tf.global_variables_initializer()

train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2000):
        sess.run(train,feed_dict={x:x_data,y:y_data})
        print(Weights_L1)
        print(biases_L1)
        print(Neurons_L1)
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()