import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'blood_eval_tmp',
                        """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                        """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/blood_train_tmp',
                        """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                         """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                         """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                      """Whether to run eval only once.""")

 def evaluate():
   """Eval blood data for a number of steps."""
   with tf.Graph().as_default() as g:
     # Get images and labels for blood data
     eval_data = FLAGS.eval_data == 'test'
     images, labels = cifar10.inputs(eval_data=eval_data)

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


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()