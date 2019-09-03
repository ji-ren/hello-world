#!/usr/bin/env python3

import tensorflow as tf

#获取数据集
#readSLHA
x_data=0
y_data=0

#批次
num_spectr=100000
batch_size=100
n_batch=num_spectr//batch_size

#两个placeholder
x=tf.placeholder(tf.float32,[None,12])
y=tf.placeholder(tf.float32,[None,2])

#第一层隐含层
Weights_L1=tf.Variable(tf.random.normal([12,32]))
biases_L1=tf.Variable(tf.zeros([1,32]))
Neurons_L1=tf.matmul(x,Weights_L1)+biases_L1
#激活第一层
L1=tf.nn.relu(Neurons_L1)
# L1=tf.sigmoid(Neurons_L1)
# L1=tf.tanh(Neurons_L1)
# L1=tf.nn.dropout(Neurons_L1)

#第二层隐含层
Weights_L2=tf.Variable(tf.random.normal([32,16]))
biases_L2=tf.Variable(tf.zeros([1,16]))
Neurons_L2=tf.matmul(L1,Weights_L2)+biases_L2
#激活第二层
L2=tf.nn.relu(Neurons_L2)
# L2=tf.sigmoid(Neurons_L2)
# L2=tf.tanh(Neurons_L2)
# L2=tf.nn.dropout(Neurons_L2)

#第三层隐含层
Weights_L3=tf.Variable(tf.random.normal([16,8]))
biases_L3=tf.Variable(tf.zeros([1,8]))
Neurons_L3=tf.matmul(L2,Weights_L3)+biases_L3
#激活第三层
L3=tf.nn.relu(Neurons_L3)
# L3=tf.sigmoid(Neurons_L3)
# L3=tf.tanh(Neurons_L3)
# L3=tf.nn.dropout(Neurons_L3)

#输出层
Weights_L4=tf.Variable(tf.random.normal([8,2]))
biases_L4=tf.Variable(tf.zeros([1,2]))
Neurons_L4=tf.matmul(L3,Weights_L4)+biases_L4
#预测
prediction=tf.nn.relu(Neurons_L4)
# prediction=tf.sigmoid(Neurons_L4)
# prediction=tf.tanh(Neurons_L4)
# prediction=tf.nn.dropout(Neurons_L4)

#代价函数
loss=tf.reduce_mean(tf.square(y-prediction))

#梯度下降法
Learning_rate=0.1
train=tf.train.GradientDescentOptimizer(Learning_rate).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#启动会话
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        for batch in range(n_batch):
            sess.run(train,feed_dict={x:x_data,y:y_data})
    #预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    
    
    
    #改进
    #消除过拟合
    #跳出局部最小值
    #激活函数，代价函数
    #其他神经网络


