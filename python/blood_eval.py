import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'blood_eval_tmp',
                        """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'blood_train_tmp',
                        """Directory where to read model checkpoints.""")

def eval_once(saver, summary_writer, top_k_op, summary_op):
   """Run Eval once.

   Args:
     saver: Saver.
     summary_writer: Summary writer.
     top_k_op: Top K op.
     summary_op: Summary op.
   """
   with tf.Session() as sess:
     ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
     if ckpt and ckpt.model_checkpoint_path:
       # Restores from checkpoint
       saver.restore(sess, ckpt.model_checkpoint_path)
       # extract global_step from it.
       global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
       print("checkpoint found at step %d", global_step)
     else:
       print('No checkpoint file found')
       return

     # Start the queue runners.
     coord = tf.train.Coordinator()
     try:
       threads = []
       for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
         threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                          start=True))

       num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
       true_count = 0  # Counts the number of correct predictions.
       total_sample_count = num_iter * FLAGS.batch_size
       step = 0
       while step < num_iter and not coord.should_stop():
         predictions = sess.run([top_k_op])
         true_count += np.sum(predictions)
         step += 1

       # Compute precision @ 1.
       precision = true_count / total_sample_count
       print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

       summary = tf.Summary()
       summary.ParseFromString(sess.run(summary_op))
       summary.value.add(tag='Precision @ 1', simple_value=precision)
       summary_writer.add_summary(summary, global_step)
     except Exception as e:  # pylint: disable=broad-except
       coord.request_stop(e)

     coord.request_stop()
     coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        blood = load_data(FLAGS.train_dir, fake_data=False, one_hot=True, dtype=tf.uint8)



        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)