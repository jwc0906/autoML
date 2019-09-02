# -*- coding: utf-8 -*-

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import random
import numpy as np

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf


def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.variance_scaling_initializer())
def bias_variable(shape):
    initial= tf.constant(0.1, shape= shape)
    return tf.Variable(initial, trainable=True)

def death_note(life_list):
    s=life_list.sum()
    r=random.randint(0, s)
    r=r-1
    for i in range(len(life_list)):
        r=r-life_list[i]
        if r<0:
            index=i
            break
    return index


"""
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b= tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)
"""

life_1_2= np.ones(2000)
life_2_3= np.ones(2000)
life_3_4= np.ones(1000)
life_4_5= np.ones(1000)
life_5_6= np.ones(100)
life_6_7= np.ones(100)

# 변수들을 설정한다.

x = tf.placeholder(tf.float32, [None, 784])
phase = tf.placeholder(tf.bool)
W1 = weight_variable("W1", [784, 2000])
b1= bias_variable([2000])
y1 = tf.nn.relu(tf.matmul(x, W1)+b1)
bat1 = tf.contrib.layers.batch_norm(y1, center=True, scale=True, is_training=phase)

W2 = weight_variable("W2", [2000, 2000])
b2= bias_variable([2000])
y2 = tf.nn.relu(tf.matmul(bat1, W2)+b2)
bat2 = tf.contrib.layers.batch_norm(y2, center=True, scale=True, is_training=phase)

W3 = weight_variable("W3", [2000, 1000])
b3= bias_variable([1000])
y3 = tf.nn.relu(tf.matmul(bat2, W3)+b3)
bat3 = tf.contrib.layers.batch_norm(y3, center=True, scale=True, is_training=phase)

W4 = weight_variable("W4", [1000, 1000])
b4= bias_variable([1000])
y4 = tf.nn.relu(tf.matmul(bat3, W4)+b4)
bat4 = tf.contrib.layers.batch_norm(y4, center=True, scale=True, is_training=phase)

W5 = weight_variable("W5", [1000, 100])
b5= bias_variable([100])
y5 = tf.nn.relu(tf.matmul(bat4, W5)+b5)
bat5 = tf.contrib.layers.batch_norm(y5, center=True, scale=True, is_training=phase)

W6 = weight_variable("W6", [100, 100])
b6= bias_variable([100])
y6 = tf.nn.relu(tf.matmul(bat5, W6)+b6)
bat6 = tf.contrib.layers.batch_norm(y6, center=True, scale=True, is_training=phase)

W7 = weight_variable("W7", [100, 10])
b7= bias_variable([10])
y7 = tf.nn.softmax(tf.matmul(bat6, W7)+b7)




# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y7), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.00003).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y7,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





# 경사하강법으로 모델을 학습한다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, phase: True})

    if i%1000==0:
        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, phase: True}))

    if i%10000==0:
        tmp_w1= sess.run(W1)
        tmp_w2= sess.run(W2)
        tmp_w3= sess.run(W3)
        tmp_w4= sess.run(W4)
        tmp_w5= sess.run(W5)
        tmp_w6= sess.run(W6)
        tmp_w7= sess.run(W7)

        #print(tmp_w7)

        #h1
        for i in range(100):
            r=death_note(life_1_2)
            life_1_2[r]=1
            tmp_w1[:,r] = np.random.uniform(-0.1, 0.1, [784])
            tmp_w2[r,:] = np.random.uniform(-0.1, 0.1, [2000])
        print(life_1_2)

        #h2
        for i in range(100):
            r=death_note(life_2_3)
            life_2_3[r]=1
            tmp_w2[:,r] = np.random.uniform(-0.1, 0.1, [2000])
            tmp_w3[r,:] = np.random.uniform(-0.1, 0.1, [1000])

        #h3
        for i in range(100):
            r=death_note(life_3_4)
            life_3_4[r]=1
            tmp_w3[:,r] = np.random.uniform(-0.1, 0.1, [2000])
            tmp_w4[r,:] = np.random.uniform(-0.1, 0.1, [1000])

        #h4
        for i in range(100):
            r=death_note(life_4_5)
            life_4_5[r]=1
            tmp_w4[:,r] = np.random.uniform(-0.1, 0.1, [1000])
            tmp_w5[r,:] = np.random.uniform(-0.1, 0.1, [100])

        #h5
        for i in range(100):
            r=death_note(life_5_6)
            life_5_6[r]=1
            tmp_w5[:,r] = np.random.uniform(-0.1, 0.1, [1000])
            tmp_w6[r,:] = np.random.uniform(-0.1, 0.1, [100])

        #h6
        for i in range(5):
            r=death_note(life_6_7)
            life_6_7[r]=1
            tmp_w6[:,r] = np.random.uniform(-0.1, 0.1, [100])
            tmp_w7[r,:] = np.random.uniform(-0.1, 0.1, [10])

        life_1_2= life_1_2+1
        life_2_3= life_2_3+1
        life_3_4= life_3_4+1
        life_4_5= life_4_5+1
        life_5_6= life_5_6+1
        life_6_7= life_6_7+1



        sess.run(tf.assign(W1, tmp_w1))
        sess.run(tf.assign(W2, tmp_w2))
        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, phase: True}))
        print("insert!!")



# 학습된 모델이 얼마나 정확한지를 출력한다.
