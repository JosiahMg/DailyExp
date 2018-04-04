# _*_ coding:utf-8 _*_
# Created time:4/3 2018
# Author : JosiahMg
# env: anaconda python 3.5

# 过拟合: 神经网络模型在训练集上的准确率高，在新的数据集预测或者分类时准确率较低，说明模型的泛化能力差
# 正则化: 在损失函数中给每个参数w加上权重，引入模型复杂度指标，从而抑制模型噪声，减少过拟合
# 使用正则化之后，损失函数loss变成了两项
#   loss = loss(y, y_) + REGULARIZER * loss(w)
# 其中第一项表示预测结果与标准答案之间的差距，如：交叉熵 均方误差 等  第二项为正则化的结果
# Tensorflow 函数实现正则化：
# 给每个w进行 tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w)
# 使用 loss = cem(标准损失函数) + tf.add_n(tf.get_collection('losses') 返回带有正则化的损失函数


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 30

# 加入正则化的w


def get_weight(shape, r_arg):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(r_arg)(w))
    return w


# 返回b
def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b

# re_anable : 是否开启正则化
# hidden_layer_sizes: 网络层个数
# 默认使用激活函数为relu


def regularization(re_enable=False, hidden_layer_sizes=11):
    seed = 2
    rdm = np.random.RandomState(seed)
    # 随机出300*2数据
    X = rdm.randn(300, 2)

    # 如果每组数据的平方和小于2则对应的标签为1  否则为0
    Y_ = [[int(x0 * x0 + x1 * x1 < 2)] for (x0, x1) in X]

    # 遍历Y_设置标签颜色 1： read   0:blue
    Y_c = [['red' if y[0] else 'blue'] for y in Y_]

    # 描点
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.show()

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    w1 = get_weight([2, hidden_layer_sizes], 0.01)
    b1 = get_bias([hidden_layer_sizes])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([hidden_layer_sizes, 1], 0.01)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2

    # 定义损失函数 不含正则化
    loss_mse = tf.reduce_mean(tf.square(y - y_))

    # 定义损失函数， 包含正则化
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    if not re_enable:
        print('未使用正则化')
        train_step = tf.train.AdamOptimizer(0.0003).minimize(loss_mse)  # 定义反向传播：不含正则化
    else:
        print('使用正则化')
        train_step = tf.train.AdamOptimizer(0.0003).minimize(loss_total)  # 定义反向传播，包含正则化

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 60000
        for i in range(steps):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                print('After {} steps, loss is:{}'.format(i, sess.run(loss_mse, feed_dict={x: X, y_: Y_})))

        # xx 在-3 到3 之间以步长为0.01 ，yy 在-3到3 之间以步长0.01，生成二维网络坐标点
        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        # 将xx, yy 拉直 并合并成一个2列的矩阵，得到一个网络坐标点的集合
        grid = np.c_[xx.ravel(), yy.ravel()]

        # 将网络坐标喂入神经网络
        probs = sess.run(y, feed_dict={x: grid})
        # probs的shape调整成xx的样子
        probs = probs.reshape(xx.shape)

        #print('w1:', sess.run(w1))
        #print('w2:', sess.run(w2))
        #print('b1:', sess.run(b1))
        #print('b2:', sess.run(b2))

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()


regularization(re_enable=True, hidden_layer_sizes=20)
