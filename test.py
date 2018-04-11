# _*_ coding:utf-8 _*_
# Created time : 3/16 2018
# Author : JosiahMg
# env: anaconda python3.6

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tf import fully_connected

'''
读取source/model目录下的模型，
读取source/MNIST_data下的数据，
统计该数据在该模型下的准确率
'''


def fully_connected_test():
    mnist = input_data.read_data_sets('./source/MNIST_data/', one_hot=True)
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, fully_connected.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, fully_connected.OUTPUT_NODE])

        y = fully_connected.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(fully_connected.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # tf.argmax(y, 1)的维度是 数据个数*1
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state('./source/model/')
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名的尾部信息取出step数目
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('After {} steps, test accuracy {}'.format(global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(5)   # 每隔5s执行一次


def main():
    fully_connected_test()


if __name__ == '__main__':
    main()








