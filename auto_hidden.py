# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


THRESHOLD=6
H1_NUM=20
H2_NUM=20


def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape,
       initializer=tf.contrib.layers.variance_scaling_initializer())
def bias_variable(shape):
    initial= tf.constant(0.1, shape= shape)
    return tf.Variable(initial, trainable=True)

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784])

W1 = weight_variable("w1", [784,1000])
b1 = bias_variable([1000])
y1 = tf.nn.relu(tf.matmul(x, W1[:, :H1_NUM]) + b1[:H1_NUM])

W2 = weight_variable("w2", [1000,1000])
b2 = bias_variable([1000])
y2 = tf.nn.relu(tf.matmul(y1, W2[:H1_NUM, :H2_NUM]) + b2[:H2_NUM])

#tf.summary.histogram("y1", y1)
#tf.summary.image('y1', tf.reshape(y1, [-1,20,1,1]), max_outputs=1)

W3 = weight_variable("w3", [1000,10])
b3 = bias_variable([10])
y3 = tf.nn.softmax(tf.matmul(y2, W3[:H2_NUM,:]) + b3)

# cross-entropy 모델을 설정한다.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y3), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# 경사하강법으로 모델을 학습한다.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



division_num_h1=0
division_num_h2=0


for i in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i>999 and i%1000==0 :
        _, y1_nodes, y2_nodes, W1_backup, W2_backup, W3_backup, b1_backup, b2_backup= sess.run([train_step, y1, y2, W1, W2, W3, b1, b2], feed_dict={x: batch_xs, y_: batch_ys})
        print("y1", y1.shape)
        print("y2", y2.shape)
    else:
        _, y1_nodes, y2_nodes= sess.run([train_step, y1, y2], feed_dict={x: batch_xs, y_: batch_ys})




    y1_node= y1_nodes[0]
    y2_node= y2_nodes[0]
    #append
    if i==0:
        y1_arr= np.array([y1_node])
        y2_arr= np.array([y2_node])
    else:
        y1_arr= np.concatenate((y1_arr, np.array([y1_node])), axis=0)
        y2_arr= np.concatenate((y2_arr, np.array([y2_node])), axis=0)


    # 평균 계산하기

    if i>999 and i%1000==0:
        y1_mean= np.mean(y1_arr[-50:,:], axis=0)
        y2_mean= np.mean(y2_arr[-50:,:], axis=0)

        #print("<y1_mean>")
        #print(y1_mean)

        #print("<y2_mean>")
        #print(y2_mean)

        h1_num_save= H1_NUM
        print(h1_num_save)
        for j in range(h1_num_save):
            if y1_mean[j]>THRESHOLD:

                division_num_h1=division_num_h1+1

                tmp= W1_backup[:,j]/2
                W1_backup[:,j]= tmp
                W1_backup[:,H1_NUM]= tmp

                tmp= b1_backup[j]/2
                b1_backup[j]= tmp
                b1_backup[H1_NUM]= tmp

                tmp= W2_backup[j,:]/2
                W2_backup[j,:]= tmp
                W2_backup[H1_NUM,:]= tmp

                H1_NUM= H1_NUM+1

                sess.run(tf.assign(W1, W1_backup))
                sess.run(tf.assign(W2, W2_backup))


                print(i, "h1",j,"th cell division!!", division_num_h1)
                break

        h2_num_save= H2_NUM
        for j in range(h2_num_save):
            if y2_mean[j]>THRESHOLD:

                division_num_h2=division_num_h2+1

                tmp= W2_backup[:,j]/2
                W2_backup[:,j]= tmp
                W2_backup[:,H2_NUM]= tmp

                tmp= b2_backup[j]/2
                b2_backup[j]= tmp
                b2_backup[H2_NUM]= tmp

                tmp= W3_backup[j,:]/2
                W3_backup[j,:]= tmp
                W3_backup[H2_NUM,:]= tmp

                H2_NUM= H2_NUM+1

                sess.run(tf.assign(W2, W2_backup))
                sess.run(tf.assign(W3, W3_backup))

                print(i, "h2",j,"th cell division!!", division_num_h2)

                break
        y1_node=0
        y2_node=0





    if i %1000==0:
        print("[",i,"]", ":", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        #print(sess.run(tf.shape(y1), feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
