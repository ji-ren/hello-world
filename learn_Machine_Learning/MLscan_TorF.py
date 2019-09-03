#!/usr/bin/env python3
import tensorflow as tf

#read
x_data=0
y_data=0    #ont-hot

num_spectr=1000000
batch_size=100
n_batch=num_spectr//batch_size

x=tf.placeholder(tf.float32,([None,12]))
y=tf.placeholder(tf.float32,([None,2]))

W1=tf.Variable(tf.zeros(12,64))
b1=tf.Variable(tf.zeros([64]))
N1=tf.matmul(x,W1)+b1
L1=tf.nn.relu(N1)

W2=tf.Variable(tf.zeros(64,32))
b2=tf.Variable(tf.zeros([32]))
N2=tf.matmul(x,W2)+b2
L2=tf.nn.relu(N2)

W3=tf.Variable(tf.zeros(32,16))
b3=tf.Variable(tf.zeros([16]))
N3=tf.matmul(x,W3)+b3
L3=tf.nn.relu(N3)

W4=tf.Variable(tf.zeros(16,8))
b4=tf.Variable(tf.zeros([8]))
N4=tf.matmul(x,W4)+b4
prediction=tf.nn.softmax(N4)

loss=-tf.reduce_sum(y*tf.log(prediction))
train=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init=tf.global_variables_initializer()
correct=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1001):
        for batch in range(n_batch):
            batch_xs,batch_ys=next_batch(batch_size)