import tensorflow as tf
import blood_data
import re
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def inputs(eval_data):
    return blood_data.inputs(fake_data=False, one_hot=True, dtype=tf.uint8, eval_data=eval_data)


def weight_variable(shape, name='generic'):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def inference(x, keep_prob):
    with tf.variable_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32], 'weights')
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1, name=scope.name)
        _activation_summary(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64], 'weights')
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name=scope.name)
        _activation_summary(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('local3') as scope:
        W_fc1 = weight_variable([21 * 21 * 64, 1024], 'weights')
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 21*21*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name=scope.name)
        _activation_summary(h_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('local4') as scope:
        W_fc2 = weight_variable([1024, 2], 'weights')
        b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv, W_conv1


def loss(y_conv, y_):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    return cross_entropy


def train(total_loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)
    return train_step


def accuracy(y_conv, y_):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    tmp_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', tmp_accuracy)
    return tmp_accuracy


def predictions(y_conv):
    return y_conv

