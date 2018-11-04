import numpy as np
import tensorflow as tf
import time
import pandas as pd


x = np.load('data.npy')  # 480x340x16
y = np.load('label.npy')  # 480x10

# a = np.arange(0, 12, dtype=np.float32)
# np.random.shuffle(a)
# print(a)
# d = pd.get_dummies(a)
# print(d)


start_time = time.time()
# input_shape
with tf.name_scope('Input'):
    x_input = tf.placeholder(tf.float32, shape=[None, 340, 16, 1], name='x_input')
    y_input = tf.placeholder(tf.float32, shape=[None, 10], name='y_input')
    pass

# conv1
with tf.name_scope('Conv1'):
    W1 = tf.Variable(tf.truncated_normal([4, 1, 1, 32], dtype=tf.float32, stddev=1e-1, name='weights'))
    W1x = tf.nn.conv2d(x_input, W1, strides=[1, 1, 1, 1], padding='SAME')
    b1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32]), name='bias')
    Wx_plus_b = tf.nn.bias_add(W1x, b1)
    conv1 = tf.nn.relu(Wx_plus_b, name='conv1')  # or Use ReLu

    print('conv.shape=', W1x.shape)
    print('bias.shape=', b1.shape)
    print('bias.shpae=', Wx_plus_b.shape)
    print('conv1_out.shape=', conv1.shape)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], name='pool1',
                           padding='SAME')  # or Max-pool
    print('pool1_out.shape=', pool1.shape)

# conv2
with tf.name_scope('Conv2'):
    W2 = tf.Variable(tf.truncated_normal([2, 1, 32, 16], dtype=tf.float32, stddev=1e-1, name='weights'))
    W2x = tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME')
    b2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[16]), name='bias')
    Wx_plus_b = tf.nn.bias_add(W2x, b2)
    conv2 = tf.nn.relu(Wx_plus_b, name='conv2')  # or Use ReLu

    print('conv.shape=', W2x.shape)
    print('bias.shape=', b2.shape)
    print('bias.shpae=', Wx_plus_b.shape)
    print('conv1_out.shape=', conv2.shape)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], name='pool2',
                           padding='SAME')  # or Max-pool
    print('pool1_out.shape=', pool2.shape)

# conv345
with tf.name_scope('Conv2'):
    W3 = tf.Variable(tf.truncated_normal([2, 1, 16, 16], dtype=tf.float32, stddev=1e-1, name='weights'))
    W3x = tf.nn.conv2d(pool2, W3, [1, 1, 1, 1], padding='SAME')
    b3 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[16]), name='bias')
    Wx_plus_b = tf.nn.bias_add(W3x, b3)
    conv3 = tf.nn.relu(Wx_plus_b, name='conv3')  # or Use ReLu

    W4 = tf.Variable(tf.truncated_normal([2, 1, 16, 16], dtype=tf.float32, stddev=1e-1, name='weights'))
    W4x = tf.nn.conv2d(conv3, W4, [1, 1, 1, 1], padding='SAME')
    b4 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[16]), name='bias')
    W4x_plus_b = tf.nn.bias_add(W4x, b4)
    conv4 = tf.nn.relu(W4x_plus_b, name='conv4')  # or Use ReLu

    W5 = tf.Variable(tf.truncated_normal([2, 1, 16, 16], dtype=tf.float32, stddev=1e-1, name='weights'))
    W5x = tf.nn.conv2d(conv4, W5, [1, 1, 1, 1], padding='SAME')
    b5 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[16]), name='bias')
    W5x_plus_b = tf.nn.bias_add(W5x, b5)
    conv5 = tf.nn.relu(W5x_plus_b, name='conv5')  # or Use ReLu

    # Local Response Normalization -> generalization -- 局部响应归一化
    norm1 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    print('norm1.shape=', norm1.shape)
    pass

# flatten
x_flatten = tf.layers.flatten(norm1)
print('x_flatten.shape=', x_flatten.shape)

# FC relu
W_fc1 = tf.Variable(tf.truncated_normal([11008, 1024], dtype=tf.float32, mean=0, stddev=1e-1), name='W_fc1')
fc1 = tf.nn.relu(tf.matmul(x_flatten, W_fc1))
print('fc.shape=', fc1.shape)

# softmax
W_s = tf.Variable(tf.truncated_normal([1024, 10], dtype=tf.float32, mean=0, stddev=1e-1), name='W_softmax')
b = tf.Variable(initial_value=tf.zeros(shape=[10]), name='bias')
res = tf.nn.softmax(tf.add(tf.matmul(fc1, W_s), b))
print('softmax.shape=', res.shape)

y_predict = res

with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y_predict))
with tf.name_scope('Train'):
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_input, 1), tf.argmax(y_predict, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(10000):
        indices = np.arange(480)
        np.random.shuffle(indices)
        # for i in range(0, 480, 20):
        # x_in = []
        # y_in = []
        # for j in range(100):
        #     tmp_x = x_in[indices[i + j]][np.newaxis, :]
        #     tmp_y = y_in[indices[i + j]]
        #     if j == 0:
        #         x_in = tmp_x
        #         y_in = tmp_y
        #     else:
        #         x_in = np.vstack((x_in, tmp_x))
        #         y_in = np.hstack((y_in, tmp_y))
        # y_in = pd.get_dummies(y_in)

        x_in = x[indices[0]]
        y_in = y[indices[0]]
        for j in range(1, 480):
            x_in = np.vstack((x_in, x[indices[j]]))
            y_in = np.hstack((y_in, y[indices[j]]))
        x_in = x_in.reshape([480, 340, 16, 1])
        y_in = y_in.reshape([480, 10])
        print(y_in.shape)

        a, b = sess.run([optimizer, cross_entropy], feed_dict={x_input: x_in[:400], y_input: y_in[:400]})
        print("\r%lf\n" % b, end='')

        acc = 0
        # for i in range(80):
        #     x_in = x[i:i + 200, :, :, :]
        #     y_in = pd.get_dummies(y_in[i:i + 200])
        acc += sess.run(accuracy, feed_dict={x_input: x_in[400:], y_input: y_in[400:]})
        print("Epoch ", epoch, 'Loss: ', b, ' validation rate: ', acc)


end_time = time.time()

print('Total time: ', end_time-start_time)
print(1)


