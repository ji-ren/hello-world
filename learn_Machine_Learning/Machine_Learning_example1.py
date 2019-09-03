#!/usr/bin/env python3
##数据代码
#导入库
import tensorflow as tf
import numpy as np
#即显模式
tf.compat.v1.enable_eager_execution()
#生成数据
n_example=100
X=tf.random.normal([n_example])
#误差
noise=tf.random.uniform([n_example],-0.5,0.5)
Y=X*2+noise
#训练集
train_x=X[:80]
train_y=Y[:80]
#测试集
test_x=X[80:]
test_y=Y[80:]

##模型代码
#用于定义参数
import tensorflow.contrib.eager as tfe
#定义模型类
class Model(object):
    def __init__(self):
        #参数初始化
        self.W=tfe.Variable(1.0)
        self.b=tfe.Variable(1.0)
    def __call__(self,x):
        #正向传递
        y=self.W*x+self.b
        return y
model=Model()

##学习代码
#定义均方差误差
def loss(prediction,label):
    loss=tf.reduce_mean(tf.square(prediction-label))
    return loss
#定义反向传递
def train(model,x,y,learning_rate,batch_size,epoch):
    #次数
    for e in range(epoch):
        #洗牌
        # r=np.random.permutation(len(x))
        # x=x[r]
        # y=y[r]
        # x=np.random.permutation(x)
        # y=np.random.permutation(x)
        #批量
        # for b in range(0,len(x.numpy()),batch_size):
        for b in range(0,len(x.numpy()),batch_size):
            # print(x)
            # print(x.numpy())
            #梯度
            with tf.GradientTape() as tape:
                loss_value=loss(model(x[b:b+batch_size]),y[b:b+batch_size])
                dW,db=tape.gradient(loss_value,[model.W,model.b])
            #更新参数
            model.W.assign_sub(dW*learning_rate)
            model.b.assign_sub(db*learning_rate)
        print('Epoch:%03d|Loss:%.3f|W:%.3f|b:%.3f' %(e,loss(model(x),y),model.W.numpy(),model.b.numpy()))

#学习
train(model,train_x,train_y,learning_rate=0.01,batch_size=2,epoch=100)
#评估
test_p=model(test_x)
print('Final Test Loss: %s' %loss(test_p,test_y).numpy())
#可视化
import matplotlib.pyplot as plt
plt.plot(test_x,test_y-noise[80:],color='red')
plt.scatter(test_x,test_y)
plt.scatter(test_x,test_p)
plt.legend(['rule','real','predicted'])
#预测
test_p=model([1,2])
print(test_p.numpy())