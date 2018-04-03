# _*_ coding:utf-8 _*_
# Created time: 4/3 2018
# Author : JosiahMg
# env: anaconda python3.5


import numpy as np
import tensorflow as tf


# 损失函数的定义
# 1、 均方误差(mse) tf.reduce_mean
# 2、自定义
# 3、交叉熵

BATCH_SIZE = 8
SEED = 23455
# 使用均方误差作为损失函数 拟合y = x1 + x2
# X : 2个特征   y:1个特征   w1 : 2*1


def loss_mse():
    # 生成数据集

    rdm = np.random.RandomState(SEED)
    X = rdm.rand(32, 2)
    Y_ = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

    # 1定义神经网络的输入  参数  输出  定义前向传播过程
    #  试图拟合 y = x1 + x2
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w1)

    # 2 定义损失函数和反向传播方法，使用均方误差和梯度下降方法
    loss_mse = tf.reduce_mean(tf.square(y_ - y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

    # 3 生成会话，训练STEPS轮

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 20000
        for i in range(steps):
            start = (i * BATCH_SIZE) % 32
            end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 500 == 0:
                print('current w1=', sess.run(w1))

        print('Final w1 is:', sess.run(w1))


#  预测某产品的生产个数，假设成本是1元 利润为9元，则预测时我们希望多预测一些
#  反之如果利润是1元 成本是9元 则我们希望少预测点产量，因为预测多了销售不出去损失的多
# 自定义损失函数  成本:COST=1   利润:PROFIT=9   y_:实际产量  y:预测产量
#  loss_self = (y>y_)? cost*(y-y_):PROFIT*(y_-y)

COST = 9
PROFIT = 1


def loss_self():
    rdm = np.random.RandomState(SEED)
    X = rdm.rand(32, 2)
    Y = [[x1+x2+(rdm.rand()/10.0 - 0.05)] for (x1, x2) in X]

    # 1定义神经网络输入  参数 输出以及前向传播过程
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w1)


    # 定义损失函数  反向传播方法
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*COST, (y_-y)*PROFIT))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # 生成会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 3000
        for i in range(steps):
            start = (i*BATCH_SIZE) % 32
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
            if i%500 == 0:
                print('After {} training steps,w1 is{}'.format(i, sess.run(w1)))

        print('Final w1 is',sess.run(w1))


# 交叉熵(Cross Entropy) 表示两个概率分布之间的距离。
# 交叉熵越大，两个概率分布距离越远，两个概率分布越相异
# 交叉熵越小，两个概率分布距离越近，两个概率分布越相似
# H(y_, y) = - sum(y_ * log y)
# Tensorflow 表示 ce = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, le-12, 1.0)))
# 对应二分类问题，已知标准答案y_=(1, 0)  则y1=(0.6, 0.4)  y2=(0.8, 0.2)哪个更和答案相似
# H1((1，0),(0.6，0.4)) = -(1*log0.6 + 0*log0.4) = 0.222
# H2((1,0),(0.8,0.2)) = -(1*log0.8+0*log0.2) = 0.097
# 可以看到y2更小，则和标准答案更接近


# 多分类问题可以使用sftemax函数处理
# Tensorflow只一般让模型输出经过sofemax函数，获得分类的概率，再与标准答案对比，求出交叉熵，得到损失函数
# 代码表示如下
#ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#cem = tf.reduce_mean(ce)

