# -*- coding: utf-8 -*-

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import matplotlib.pyplot as plt
import numpy as np

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf

def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.variance_scaling_initializer())
def bias_variable(shape):
    initial= tf.constant(0.1, shape= shape)
    return tf.Variable(initial, trainable=True)

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784])

W1 = weight_variable("w1", [784,20])
b1 = bias_variable([20])
y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

W2 = weight_variable("w2", [20,20])
b2 = bias_variable([20])
y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)

#tf.summary.histogram("y1", y1)
#tf.summary.image('y1', tf.reshape(y1, [-1,20,1,1]), max_outputs=1)

W3 = weight_variable("w3", [20,10])
b3 = bias_variable([10])
y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)

# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y3), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merge = tf.summary.merge_all()

# 경사하강법으로 모델을 학습한다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter('./tensorboard', sess.graph)
global_step=0



for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i %1000==0:
        print(i, ":", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        #print(sess.run(tf.shape(y1), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

for i in range(200):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i==0:
        img1= np.array(sess.run(y1, feed_dict={x: batch_xs, y_: batch_ys})[0]).reshape((1,20))

        print(img1.shape)
    else:
        img1= np.concatenate((img1, np.array([sess.run(y1, feed_dict={x: batch_xs, y_: batch_ys})[0]]).reshape((1,20))), axis=0)




print(i, ":", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

fig = plt.figure()

subplot = fig.add_subplot(1, 6, 1)
subplot.set_xticks([])
subplot.set_yticks([])
subplot.imshow(img1, cmap=plt.cm.gray_r)

print("<",i,">")
print(img1)
m_array=[]
v_array=[]
for i in range(20):
    print("<"+str(i)+">")
    m= np.mean(img1[:,i])
    v= np.var(img1[:,i])
    print("mean:", m)
    print("var:", v)
    m_array.append(m)
    v_array.append(v)

plt.show()

plt.figure()
plt.scatter(m_array, v_array)
plt.show()
