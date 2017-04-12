import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
decay = 0.9 # discounted factor of learning rate
training_epochs = 10
batch_size = 100
conv_drop_rate_value = 0.8
fully_drop_rate_value = 0.5

X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # 몇개 올지는 모르고, 28 by 28 이미지의 1 color(흑백)
Y = tf.placeholder(tf.float32, [None, 10])

conv_drop_rate = tf.placeholder(tf.float32)
fully_drop_rate = tf.placeholder(tf.float32)

script_dir = os.path.dirname(os.path.abspath(__file__))
mnist = input_data.read_data_sets(script_dir + "/../mnist/data/", one_hot=True)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # 3*3 patch, depth 1 (color), 32개
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
# 이미지의 사이즈가 처음엔 28인데, max pooling하면서 2로 나뉘어짐
# 28 -> 14 -> 7 -> 3.5인데, 반올림 4
W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))

with tf.name_scope('layer1') as scope :
    # L1 Conv shape = (?, 28, 28, 32)
    # Pool -> (? 14, 14, 32)
    L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'))
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' ) # 2*2 pooling patch
    L1 = tf.nn.dropout(L1, conv_drop_rate)
with tf.name_scope('layer2') as scope :
    # L2 Conv shape = (?, 14, 14, 64)
    # Pool -> (?, 7, 7, 64)
    L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME'))
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    L2 = tf.nn.dropout(L2, conv_drop_rate)
with tf.name_scope('layer3') as scope :
    # L3 Conv shape = (>, 7, 7, 128)
    # Pool -> (?, 4, 4, 128)
    # Reshape -> (?, 2048)
    L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME'))
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    L3 = tf.reshape(L3, [-1, W4.get_shape().as_list()[0]])
    # w4가 [128*4*4, 625] 니까, 여기서 0번째 원소가 크기 !
    L3 = tf.nn.dropout(L3, conv_drop_rate)
with tf.name_scope('fully connected layer') as scope :
    # L4 FC 4*4*128 inputs -> 625 outputs
    L4 = tf.nn.relu(tf.matmul(L3, W4))
    L4 = tf.nn.dropout(L4, fully_drop_rate)

# L4 : 1 by 625
# W5 : 625 by 10
# output : 1 by 10
# 즉, 10개의 숫자에 대한 확률값들
model = tf.matmul(L4, W5)

with tf.name_scope('cost') as scope :
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))

train = tf.train.RMSPropOptimizer(learning_rate, decay).minimize(cost)

init = tf.global_variables_initializer()

with tf.Seesion() as sess :
    sess.run(init)
    test_feed_dict = {
        X: mnist.test.images.reshape(-1, 28, 28, 1), # -1은 갯수가 정해지지 않아, 모를 경우.. None 써도되나?
        Y: mnist.test.labels,
        conv_drop_rate: conv_drop_rate_value,
        fully_drop_rate: fully_drop_rate_value
    }

    for epoch in range(training_epochs) :
        total_batch = int(mnist.train.num_examples / batch_size)

        for step in range(total_batch) :
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            feed_dict = {
                X : batch_xs.reshape(-1, 28, 28, 1),
                Y : batch_ys,
                conv_drop_rate : conv_drop_rate_value,
                fully_drop_rate : fully_drop_rate_value
            }

            sess.run(train, feed_dict = feed_dict)

        check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)

        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
