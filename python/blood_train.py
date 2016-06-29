import tensorflow as tf
import blood_model
import os
import numpy as np


FLAGS = tf.app.flags.FLAGS
RUN = 'all_five_cells'
tf.app.flags.DEFINE_string('checkpoint_dir', RUN+'/blood_train_tmp',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_string('summaries_dir', RUN+'/blood_eval_tmp',
                           """Summaries directory""")
tf.app.flags.DEFINE_string('max_steps', 20000,
                           """Maximum steps to train the model""")
tf.app.flags.DEFINE_string('continue_run', False,
                           """Continue from when training stopped?""")
tf.app.flags.DEFINE_string('batch_size', 64,
                           """batch size""")


def train():
    """Train blood_model for a number of steps."""

    # declare placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 75, 75, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 5])
        tf.image_summary('input', x, 50)
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)

    # Get images and labels for blood_model.
    blood_datasets = blood_model.inputs(eval_data=False)
    # build the convolution network
    conv_output, _, _, _, _ = blood_model.inference(x, keep_prob)
    # Calculate loss.
    loss = blood_model.loss(conv_output, y_)
    accuracy = blood_model.accuracy(conv_output, y_)

    train_op = blood_model.train(loss)

    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()

    check_filesystem()

    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    validation_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/validation', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)

    global_step = reload_checkpoint_if_exists(sess, saver, train_writer, validation_writer, test_writer)

    for step in range(global_step+1, FLAGS.max_steps):
        batch = blood_datasets.train.next_batch(FLAGS.batch_size)
        _, loss_output = sess.run([train_op, loss], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        assert not np.isnan(loss_output)
        if step % 100 == 0:
            summary, train_accuracy = sess.run([summary_op, accuracy], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_writer.add_summary(summary, step)
            print("step %d, training accuracy %g" % (step, train_accuracy))

        if (step % 1000 == 0 or (step + 1) == FLAGS.max_steps) and not step == 0:
            summary_validation, accuracy_validation = sess.run([summary_op, accuracy], feed_dict={
                    x: blood_datasets.validation.images, y_: blood_datasets.validation.labels, keep_prob: 1.0})
            validation_writer.add_summary(summary_validation, step)
            print("validation accuracy %g" % accuracy_validation)

            summary_test, accuracy_test = sess.run([summary_op, accuracy], feed_dict={
                    x: blood_datasets.testing.images, y_: blood_datasets.testing.labels, keep_prob: 1.0})
            test_writer.add_summary(summary_test, step)
            print("test accuracy %g" % accuracy_test)

            # save checkpoint
            checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print("saving checkpoint")


def check_filesystem():
    if FLAGS.continue_run:
        # start a new run, set flag to continue, so there is nothing
        # check if something there, if not, create, but don't delete
        if not tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.MakeDirs(FLAGS.summaries_dir)
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'train'))
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'validation'))
            tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'test'))
        if not tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    else:
        # delete checkpoints and event summaries because training restarted
        if tf.gfile.Exists(FLAGS.summaries_dir):
            tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(FLAGS.summaries_dir)
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'train'))
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'validation'))
        tf.gfile.MakeDirs(os.path.join(FLAGS.summaries_dir, 'test'))
        if tf.gfile.Exists(FLAGS.checkpoint_dir):
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)


def reload_checkpoint_if_exists(sess, saver, train_writer, validation_writer, test_writer):
    global_step = -1
    if FLAGS.continue_run:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
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
    return global_step


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()