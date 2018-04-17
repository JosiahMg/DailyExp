# _*_ coding:utf-8 _*_

# tf.get_colletction('') 从集合中取全部变量，生成一个列表
# tf.add_n([])   列表内对应元素相加
# tf.cast(x, dtype)  把x转为dtype类型
# tf.argmax(x, axis)  返回最大值所在索引号，如：tf.argmax([1, 0, 0], 1) 返回0
# tf.equal(A, B) : 两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A相同

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# 输入特征个数，即像素点个数 28*28
INPUT_NODE = 784
# 输出特征个数，即0-9 共十个数字
OUTPUT_NODE = 10
# 第一个隐藏层节点个数
LAYER1_NODE = 500


BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = '..\source\model'
MODEL_NAME = 'mnist_model_fully_connected'
ROOT_PATH = os.path.abspath('..')

# 定义前向传播


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        # L2正则化加入到'losses'列表中
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])

    # 由于输出需要经过softmax函数，y必须符合正态分布，因此不能使用relu函数进行非线性转换。
    y = tf.matmul(y1, w2) + b2
    return y


# 定义反向传播


def backward(mnist):
    # 源输入特征
    x = tf.placeholder(tf.float32, [None, INPUT_NODE])
    # 源特征标签
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
    # 预测的特征标签
    y = forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    # tf.argmax(y_, 1) 将y_中最大值的索引返回出来，比如：[0,1,0,0,0,0,0,0,0,0] 返回值为1
    # 第二个参数的含义是对y_中第一个列表取最大值索引
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续传，异常终止训练后不用重启训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Continue train from ',ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i%1000 == 0:
                print('After {} training step(s),loss on training batch is {}'.format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist_data = input_data.read_data_sets('../source/MNIST_data/', one_hot=True)
    backward(mnist_data)

# 使用预测后的模型进行手写图片的预测
# img_arr : 1 * 784
# model_path : 模型路径


def restore_model(img_arr, model_path):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, INPUT_NODE])
        y = forward(x, None)
        pre_value = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pre_value = sess.run(pre_value, feed_dict={x: img_arr})
                return pre_value
            else:
                print('No checkpoint file found')
                return -1







if __name__ == '__main__':
    main()
