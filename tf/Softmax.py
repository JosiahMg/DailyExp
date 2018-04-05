# _*_ coding:utf-8 _*_
# Created time : 4/5 2018
# Author : JosiahMg
# env : anaconda python 3.5

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt

# 从tensorflow中获取手写图片集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# mnist.train.images.shape = (55000, 784)
# 表示有55000张图片，每个图片有784 = 28*28 个像素点
# 显示第13张图片
pic = mnist.train.images[12].reshape(28, 28)
plt.imshow(pic, cmap='gray')
plt.show()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for _ in range(1000):
        # 每次取100张图片 batch[0]存放图片信息，所以batch[0].shape = (100, 784)
        # batch[1] 存放对应的标签  batch[1].shape = (100, 10)
        batch = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

    # 测试集上测试准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


