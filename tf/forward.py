# _*_ coding:utf-8 _*_
# Created time : 4/2 2018
# Author : JosiahMg
# env : anaconda python 3.5


import tensorflow as tf


# 前向传播基本结构


def forward_sample():
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


# 模块化设计


# 定义神经网络的输入 参数 输出 定义前向传播过程
# shape: w的形状，使用类别形式，如 [2, 4]
# regularizer: 正则化权重，返回值为预测或者分类结果y


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将参数w正则化损失加到总损失losses中
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 生成形状为shape的值全为0.01的b


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


# 前向传播模块化程序


def forward(x, regularizer):
    w1 = get_weight([2, 11],regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2

    return y


