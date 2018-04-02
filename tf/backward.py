# _*_ coding:utf-8 _*_
# Created time: 4/2 2018
# Author : JosiahMg
# env: anaconda python-3.5

import numpy as np
import tensorflow as tf


BATCH_SIZE = 8


def backward():

    seed = 23455
    # 随机32组数据，且每组的数据和小于1则对应的表情y为1 否则为0
    rmd = np.random.RandomState(seed)
    X = rmd.rand(32, 2)
    Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

    #  定义神经网络的输入和前向传播过程

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 定义损失函数以及反向传播方法

    loss_mse = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
    # train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss_mse)
    # train_step = tf.train.AdamOptimizer(0.001).minimize(loss_mse)

    # 生成会话，训练STEPS轮

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('w1:', sess.run(w1))
        print('w2:', sess.run(w2))
        print('\n')

        steps = 3000
        for i in range(steps):
            start = (i * BATCH_SIZE) % 32
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 500 == 0:
                total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
                print('After {} training steps,loss_mse on all data is {}'.format(i, total_loss))

        print('\n')
        print('w1:', sess.run(w1))
        print('w2:', sess.run(w2))


backward()