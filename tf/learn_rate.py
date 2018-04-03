# _*_ coding:utf-8 _*_
# Created time: 4/3 2018
# Author: JosiahMg
# env: anaconda python=3.5

# 学习率表示每次参数更新的速率，学习率过大会导致优化参数在最小值之间波动，不收敛，过小导致收敛速度慢
# 学习率的设置：恒定学习率以及指数衰减学习率
# w(n+1)  =  w(n) - learning_rate * 梯度
# 假设损失函数loss = (w+1)**2  则梯度为 2w+2   如果参数初始值为5，学习率为0.2  则参数和损失函数更新如下：
# 1    w:5       5-0.2*(2*5+2) = 2.6
# 2    w:2.6     2.6-0.2*(2*2.6+2) = 1.16
# 3    w:1.16     1.16-0.2*(2*1.16.6+2) = 0.296
# 4    w:0.296    ....


import tensorflow as tf

# loss = (w+1)**2  使用恒定学习率迭代损失函数使得最小
# rate:学习率
# steps:迭代次数


def constant_learn_rate(rate, steps):
    # 初始化w=5
    w = tf.Variable(tf.constant(5, dtype=tf.float32))

    # 定义损失函数  loss = (w+1)**2
    loss = tf.square(w+1)

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(rate).minimize(loss)

    # 生成会话

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(steps):
            sess.run(train_step)
            print('After {} steps w:{}  loss {}'.format(i, sess.run(w), sess.run(loss)))


# 指数衰减学习率，学习率随着训练轮数变化而更新


LEARNING_RATE_BASE = 0.1   # 初始化学习率
LEARNING_RATE_DECAY = 0.99 # 学习率衰减率
LEARNING_RATE_STEP = 1     # 多少轮BATCH_SIZE后更新一次学习率

# steps:迭代次数
# 指数衰减学习率
# global_step的作业是什么？


def exponential_learn_rate(steps):
    # trainable设置为不训练  训练次数计算器,初始值为0
    global_step = tf.Variable(0, trainable=False)

    # 定义指数下降学习率
    # staircase:为True表示学习率阶梯型衰减     False表示学习率平滑下降
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)

    # 定义待优化参数，初始值为5
    w = tf.Variable(tf.constant(5, dtype=tf.float32))

    # 定义损失函数
    loss = tf.square(w + 1)

    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 生成会话

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(steps):
            sess.run(train_step)
            print(
                'After {} steps global_step {} w {} learn_rate {} loss {}'.format(i, sess.run(global_step), sess.run(w),
                                                                                  sess.run(learning_rate),
                                                                                  sess.run(loss)))

