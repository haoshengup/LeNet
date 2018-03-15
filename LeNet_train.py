import tensorflow as tf
from tensor_flow import LeNet_inference
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

BATCH_SIZE = 100
MAX_STEP = 3001
LEARNING_RATE = np.exp(0.01)
LEARNING_RATE_DECAY = 0.99
MOVING_DECAY = 0.99
REGULARIZATION_RATE = 0.0001

MODEL_PATH = 'LeNet_path/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet_inference.IMAGE_SIZE, \
                                    LeNet_inference.IMAGE_SIZE, LeNet_inference.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet_inference.OUTPUT_NODE])

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet_inference.inference(x,train, regularizer)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE, global_step=global_step, \
                                               decay_steps=mnist.train.num_examples//BATCH_SIZE, decay_rate=LEARNING_RATE_DECAY)
    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entroy_mean = tf.reduce_mean(cross_entroy)
    losses = cross_entroy_mean + tf.add_n(tf.get_collection('loss'))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses, global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op('train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(MAX_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshape = np.reshape(xs, (BATCH_SIZE, LeNet_inference.IMAGE_SIZE, \
                                    LeNet_inference.IMAGE_SIZE, LeNet_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, losses, global_step], feed_dict={x:xs_reshape, y_:ys})
            if i%300 == 0:
                print('{} step, loss: {}'.format(step, loss_value))
                saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step)

def main(argv=None):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
