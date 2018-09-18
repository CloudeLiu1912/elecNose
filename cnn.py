import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn
import os
from datetime import datetime
from tensorflow.examples.tutorials.mnist import input_data


# 将传入的label转换成one hot的形式。
def getOneHotLabel(label, depth):
    m = np.zeros([len(label), depth])
    for i in range(len(label)):
        m[i][label[i]] = 1
    return m


# 建立神经网络。
def alexnet(image, keepprob=0.5):

    # 定义卷积层1，卷积核大小，偏置量等各项参数参考下面的程序代码，下同。
    with tf.name_scope("conv1") as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(image, kernel, [1, 4, 4, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

        pass

    # LRN层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn1")

    # 最大池化层
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool1")

    # 定义卷积层2
    with tf.name_scope("conv2") as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        pass

    # LRN层
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")

    # 最大池化层
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")

    # 定义卷积层3
    with tf.name_scope("conv3") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        pass

    # 定义卷积层4
    with tf.name_scope("conv4") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        pass

    # 定义卷积层5
    with tf.name_scope("conv5") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        pass

    # 最大池化层
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool5")

    # 全连接层
    flatten = tf.reshape(pool5, [-1, 6*6*256])

    weight1 = tf.Variable(tf.truncated_normal([6*6*256, 4096], mean=0, stddev=0.01))

    fc1 = tf.nn.sigmoid(tf.matmul(flatten, weight1))

    dropout1 = tf.nn.dropout(fc1, keepprob)

    weight2 = tf.Variable(tf.truncated_normal([4096, 4096], mean=0, stddev=0.01))

    fc2 = tf.nn.sigmoid(tf.matmul(dropout1, weight2))

    dropout2 = tf.nn.dropout(fc2, keepprob)

    weight3 = tf.Variable(tf.truncated_normal([4096, 10], mean=0, stddev=0.01))

    fc3 = tf.nn.sigmoid(tf.matmul(dropout2, weight3))

    return fc3


def GetEvery100(path='/Users/Epilo/Desktop/COLLEGE/科研/石斛补充实验/', flag=False):
    e100 = []
    if flag:
        for i in range(10):
            for root, dirs, files in os.walk(path + str(i + 1), topdown=True):
                for name in files:
                    print(os.path.join(root, name))
                    df = pd.read_table(os.path.join(root, name), delimiter='\t', encoding='latin1', dtype=float,
                                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])

                    for k in range(round(len(df) / 100)):
                        print(i, '\t', k*100)
                        e100 = np.hstack((e100, df.iloc[k*100]))

        # e100 = pd.DataFrame(e100)
        # e100.to_csv('e100.csv')
    # else:
        # e100 = pd.read_csv('e100.csv')

    print('Loaded Successfully')
    # print(e100)
    # print(e100.shape)
    # print(e100.__class__)
    return e100


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     result = sess.run(op)
#     print(result)

ev100 = GetEvery100(flag=True)
print(ev100.shape)
ev100 = ev100.reshape([10, 48, 340, 16])
print(ev100.shape)
np.save('ev100.npy', ev100)

