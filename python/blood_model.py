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
    return blood_data.inputs_balanced(fake_data=False, one_hot=True, dtype=tf.uint8, eval_data=eval_data)


def prepare_input(x):
    output_list = []
    for i in range(x.get_shape().as_list()[0]):
        distorted_image = tf.random_crop(x[i, :, :, :], [75, 75, 3])
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        # Because these operations are not commutative, consider randomizing the order their operation.
        #distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        #distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_whitening(distorted_image)
        #if i == 0:
            #print(distorted_image.get_shape().as_list())
            #output_x = tf.expand_dims(distorted_image, -1)  # [75, 75, 3, 1]
            #print(output_x.get_shape().as_list())
        #else:
            #print(output_x.get_shape().as_list())
            #output_x = tf.concat(3, [output_x, distorted_image])  # [75, 75, 3, 1+1]
        output_list.append(distorted_image)
    return tf.pack(output_list)
    #return tf.transpose(output_x, perm=[3, 0, 1, 2])


def weight_variable(shape, name='generic', wd=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    var = tf.Variable(initial, name)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


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
        W_conv1 = weight_variable([5, 5, 3, 32], 'weights', wd=0.0)
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1, name=scope.name)
        _activation_summary(h_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64], 'weights', wd=0.0)
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name=scope.name)
        _activation_summary(h_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    with tf.variable_scope('local3') as scope:
        W_fc1 = weight_variable([19 * 19 * 64, 1024], 'weights', wd=0.004)
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 19*19*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name=scope.name)
        _activation_summary(h_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('local4') as scope:
        W_fc2 = weight_variable([1024, 5], 'weights', wd=0.0)
        b_fc2 = bias_variable([5])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv, W_conv1, W_conv2, h_conv1, h_conv2


def loss(y_conv, y_):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    tf.add_to_collection('losses', cross_entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


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

