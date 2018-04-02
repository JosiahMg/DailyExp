# _*_ coding:utf-8 _*_
# Created time : 4/2 2018
# Author : JosiahMg
# env : anaconda python 3.5


import tensorflow as tf


# 前向传播基本结构


def forward_model():
    # 定义输入和参数
    # 使用 placeholder定义多组数组输入
    # x 特征输入   w1 第一层神经网络   w2 第二层神经网络

    x = tf.placeholder(tf.float32, shape=(None, 2))
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 定义前向传播过程

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 调用会话计算结果

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print('y output is:', sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
        print('w1 is:', sess.run(w1))
        print('w2 is:', sess.run(w2))


