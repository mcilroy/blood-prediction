import tensorflow as tf
import blood_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

FLAGS = tf.app.flags.FLAGS
RUN = 'run2'
tf.app.flags.DEFINE_string('checkpoint_dir', RUN+'/blood_train_tmp',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('batch_size', 50,
                           """batch size""")


def show_hard_images(images_used, batch_predictions):

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 2),
                     axes_pad=0.1,)
    count = 0
    for i, val in enumerate(batch_predictions):
        if 0.4 <= val[1] <= 0.6:
            grid[count].imshow(images_used[i])
            count += 1
        if count >= 10:
            break
    print("confusing image count " + str(count))
    plt.show()


def show_misclassified_images(images_used, batch_predictions, labels):
    correct_predictions = np.equal(np.argmax(batch_predictions, 1), np.argmax(labels, 1))

    fig = plt.figure()
    fig.suptitle('left: (predict:mono, actual: neutro), right: (predict:neutro, actual: mono)', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 2),
                     axes_pad=0.1,)
    count_neutro = 0
    count_mono = 0
    for i, val in enumerate(correct_predictions):
        if not val:
            if labels[i, 1] == 1:  # neutrophile
                if count_neutro < 5:
                    grid[(count_neutro*2)+1].imshow(images_used[i])
                    count_neutro += 1
            else:
                if count_mono < 5:
                    grid[count_mono*2].imshow(images_used[i])
                    count_mono += 1
        if count_neutro >= 4 and count_mono >= 4:
            break
    #print("mislabelled image count " + str(count_mono+count_neutro))
    plt.show()


def show_filters(filters):
    filters = np.rollaxis(filters, 3, 0)
    fig = plt.figure()
    fig.suptitle('filters 1st layer', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=((filters.shape[0]/2)+1, 2),
                     axes_pad=0.1,)
    for i in range(filters.shape[0]):
        filter = filters[i, :, :, 0]
        grid[i].imshow(filter)
    print("filter # : " + str(filters.shape[0]))
    plt.show()


def eval_once():
    pass


def evaluate():
    """Train blood_model for a number of steps."""

    # declare placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, 81, 81, 3])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        tf.image_summary('input', x, 50)
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)

    # Get images and labels for blood_model.
    conv_output, W_conv1 = blood_model.inference(x, keep_prob)
    predictions = blood_model.predictions(conv_output)

    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # extract global_step from it.
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("checkpoint found at step %d", global_step)
    else:
        print('No checkpoint file found')
        return
    # foo = sess.graph.get_tensors()
    # blah = sess.graph.get_operations()
    # weights = sess.graph.get_operation_by_name('conv1/Variable')
    # weights = sess.run(weights)
    # show_hard_images(batch_val[1], predictions)
    blood_datasets = blood_model.inputs(eval_data=True)

    filters = sess.run(W_conv1)
    show_filters(filters)

    batch_val = blood_datasets.validation.next_batch_untouched(FLAGS.batch_size)

    predictions = sess.run(predictions, feed_dict={x: batch_val[0], y_: batch_val[2], keep_prob: 1.0})



    #show_misclassified_images(batch_val[1], predictions, batch_val[2])


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
