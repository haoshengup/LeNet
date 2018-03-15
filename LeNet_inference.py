import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_SIZE = 5
CONV1_DEEP = 32

CONV2_SIZE = 5
CONV2_DEEP = 64

FC_NODE = 512

def inference(x, train, regularizer):
    with tf.variable_scope('layer1_conv'):
        weights = tf.get_variable('weights', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.))
        conv1 = tf.nn.conv2d(x, weights, [1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias))
    with tf.name_scope('layer2_pool'):
        pool1 = tf.nn.max_pool(relu1, [1,2,2,1], [1,2,2,1], padding='SAME')
    with tf.variable_scope('layer3_conv'):
        weights = tf.get_variable('weights', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], \
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.))
        conv2 = tf.nn.conv2d(pool1, weights, [1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias))
    with tf.name_scope('layer4_pool'):
        pool2 = tf.nn.max_pool(relu2, [1,2,2,1], [1,2,2,1], padding='SAME')
        pool2_shape = pool2.get_shape().as_list()
        node = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
        pool2_reshape = tf.reshape(pool2, [pool2_shape[0], node])
    with tf.variable_scope('layer5_fc'):
        weights = tf.get_variable('weights', [node, FC_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer:
            tf.add_to_collection('loss', regularizer(weights))
        bias = tf.get_variable('bias', [FC_NODE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool2_reshape, weights) + bias)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    with tf.variable_scope('layer6_fc'):
        weights = tf.get_variable('weights', [FC_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer:
            tf.add_to_collection('loss', regularizer(weights))
        bias = tf.get_variable('bias', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, weights) + bias
    return logit