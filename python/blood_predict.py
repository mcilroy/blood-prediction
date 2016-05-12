import tensorflow as tf
import numpy as np
from blood_data import load_data
import os.path
import re
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'blood_train_tmp',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('summaries_dir', 'blood_eval_tmp',
                           """Summaries directory""")
tf.app.flags.DEFINE_string('max_steps', 20000,
                           """Maximum steps to train the model""")
tf.app.flags.DEFINE_string('restart', False,
                           """Restart or continue from when training stopped?""")
tf.app.flags.DEFINE_string('batch_size', 50,
                           """batch size""")
TOWER_NAME = 'tower'
blood = load_data(FLAGS.train_dir, fake_data=False, one_hot=True, dtype=tf.uint8)

sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 81, 81, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 2])
    tf.image_summary('input', x, 50)
    keep_prob = tf.placeholder(tf.float32)
    tf.scalar_summary('dropout_keep_probability', keep_prob)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


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


def show_hard_images(images_used, batch_predictions):

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 2),
                     axes_pad=0.1,)
    count = 0
    for i, val in enumerate(batch_predictions):
        if 0.4 <= val[1] <= 0.6:
            #imgplot = plt.imshow(images_used[i, :, :, :])
            grid[count].imshow(images_used[i])
            count = count + 1
        #else:
            #imgplot = plt.imshow(images_used[i, :, :, :])
        if count >= 10:
            break
    plt.show()




    # hard_images = []
    # for i, val in enumerate(batch_predictions):
    #     if 0.4 <= val[1] <= 0.6:
    #         if hard_images == []:
    #             hard_images = images_used[i, :, :, :]
    #         else:
    #             if hard_images.ndim == 4:
    #                 hard_images = np.concatenate(
    #                     [hard_images, images_used[None, i, :, :, :]])
    #             else:
    #                 hard_images = np.concatenate(
    #                     [hard_images[None, :, :, :], images_used[None, i, :, :, :]])
    #
    # fig = plt.figure(1, (4., 4.))
    # grid = ImageGrid(fig, 111, nrows_ncols=(int((hard_images.shape[0]/2)+1), 2),
    #                  axes_pad=0.1,)
    # for i in range(hard_images.shape[0]):
    #     grid[i].imshow(hard_images[i])
    # plt.show()
    # return hard_images

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([21 * 21 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 21*21*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predictions = y_conv

tf.scalar_summary('accuracy', accuracy)

sess.run(tf.initialize_all_variables())

# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
validation_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/validation', sess.graph)
test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)
saver = tf.train.Saver()

global_step = -1
if not FLAGS.restart:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # extract global_step from it.
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("checkpoint found at step %d", global_step)
        # ensure that the writers ignore saved summaries that occurred after the last checkpoint but before a crash
        train_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step)
        validation_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step)
        test_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step)
    else:
        print('No checkpoint file found')
else:
    # delete checkpoints and event summaries because training restarted
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

for step in range(global_step+1, FLAGS.max_steps):
    batch = blood.train.next_batch(FLAGS.batch_size)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if step % 100 == 0:

        batch_val = blood.validation.next_batch_untouched(FLAGS.batch_size)
        predictions = sess.run(predictions, feed_dict={
            x: batch_val[0], y_: batch_val[2], keep_prob: 1.0})
        show_hard_images(batch_val[1], predictions)

        summary, train_accuracy = sess.run([summary_op, accuracy], feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        train_writer.add_summary(summary, step)
        print("step %d, training accuracy %g" % (step, train_accuracy))

        #imgplot = plt.imshow(images[0, :, :, :])
        #imgplot2 = plt.imshow(images[1, :, :, :])

    if (step % 1000 == 0 or (step + 1) == FLAGS.max_steps) and not step == 0:
        summary_validation, accuracy_validation = sess.run([summary_op, accuracy], feed_dict={
                x: blood.validation.images, y_: blood.validation.labels, keep_prob: 1.0})
        validation_writer.add_summary(summary_validation, step)
        print("validation accuracy %g" % accuracy_validation)

        summary_test, accuracy_test = sess.run([summary_op, accuracy], feed_dict={
                x: blood.testing.images, y_: blood.testing.labels, keep_prob: 1.0})
        test_writer.add_summary(summary_test, step)
        print("test accuracy %g" % accuracy_test)

        # save checkpoint
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)




# to visualize output of poor performing cells
    # get predictions of each image in a batch
        # predictions = y_conv (batch_size, 2)
        # if predictions are bad put in bad_predictions
    # tf.image_summary('bad_predictions', bad_predictions, 50)
    # summary_bad = sess.run(bad_predictions_op, feed_dict={
    # x: blood_validation.images, y: blood_validation.labels, keep_prob: 1.0})
    # bad_writer.add_summary(summary_bad, step)
