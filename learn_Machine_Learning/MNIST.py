#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#获得数据集
mnist=input_data.read_data_sets('mnist_data',one_hot=True)

#每个批次的大小
batch_size=100
#批次的数量
n_batch=mnist.train.num_examples//batch_size

#定义两个占位符
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

#一个简单的神经网络输出层
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,W)+b)


#准备使用交叉熵
# W=tf.Variable(tf.random.normal([784,10]))
# b=tf.Variable(tf.zeros([10]))
# N0=tf.nn.sigmoid(tf.matmul(x,W)+b)

#加一层
# W1=tf.Variable(tf.random.normal([10,10]))
# b1=tf.Variable(tf.zeros([10]))
# prediction=tf.matmul(N0,W1)+b1

#代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
# loss=-tf.reduce_sum(y*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
# loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction)
#梯度下降法
train=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#正确率
correct=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

#建立会话
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(101):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:batch_xs,y:batch_ys})
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+',Testing Accuract'+str(acc))
