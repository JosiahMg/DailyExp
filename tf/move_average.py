# _*_ coding:utf-8 _*_
# Create Time : 4/4 2018
# Author : JosiahMg
# env : anaconda python 3.5

import tensorflow as tf

# 滑动平均： 记录一段时间内模型中所有参数w和b各种的平均值。利用滑动平均值可以增强模型的泛化能力
# 滑动平均值计算公式：影子 = 衰减率*影子 + （1-衰减率）* 参数
# 其中，衰减率 = min{MOVING_AVERAGE_DECAY, (1+轮数)/(10+轮数)} ，影子初始值= 参数初始值
# 用tensorflow 函数表示：
# ema = tf.train.ExponentiaMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# 其中 MOVING_AVERAGE_DECAY表示滑动平均衰减率，一般会赋值接近1的值，global_step表示当前训练轮数
# ema_op = ema.apply(tf.trainable_variables())
# ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所以待训练参数汇总为列表

MOVING_AVERAGE_DECAY = 0.99


def moving_average():
    # 1. 定义变量及滑动平均类
    # 定义一个32位浮点变量，初始值为0.0  这个代码就是不断更新w1参数，优化w1参数，滑动平均做了个w1的影子
    w1 = tf.Variable(0, dtype=tf.float32)

    # 定义NN的迭代次数，初始值为0，不可被优化(训练)
    global_step = tf.Variable(0, trainable=False)

    # 实例化滑动平均值，给衰减率0.99 当前轮数global_step
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # ema_op = ema.apply()中是更新列表，每次运行sess.run(ema_op)时，会对更新列表的元素求滑动平均值
    # 实际使用时使用tf.trainable_variables自动将所以待训练的参数汇总为列表
    ema_op = ema.apply(tf.trainable_variables())

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # ema.average(w1)获取w1的滑动平均值
        # 打印w1和w1的滑动平均值
        print(sess.run([w1, ema.average(w1)]))

        # 参数w1的值赋值为1
        sess.run(tf.assign(w1, 1))
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        # 更新step和w1的值，模拟出100轮迭代后，参数w1变成10
        sess.run(tf.assign(global_step, 100))
        sess.run(tf.assign(w1, 10))
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        # 每次sess.run会更新一次w1的滑动平均值
        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))

        sess.run(ema_op)
        print(sess.run([w1, ema.average(w1)]))


moving_average()


