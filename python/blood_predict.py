import tensorflow as tf
import blood_model
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

FLAGS = tf.app.flags.FLAGS
RUN = 'all_five_cells_balanced_paul_sameseed'
tf.app.flags.DEFINE_string('checkpoint_dir', RUN+'/checkpoints', """Directory where to read model checkpoints.""")


def display_predictions(data, predictions):
    images = [[] for i in range(5)]
    for i, pred in enumerate(predictions):
        images[pred].append(data[i])
    cols = ['Ne', 'Mo', 'Ba', 'Eo', 'Ly']
    lengths = [len(w) for w in images]
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(5, len(cols)), axes_pad=0.1,)
    for i, image_type in enumerate(images):
        for j, image in enumerate(image_type):
            if j >= 5:
                break
            grid[(len(images)*j)+i].imshow(image)

    pad = 5  # in points
    for ax, col in zip(grid.axes_all, cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large',
                    ha='center', va='baseline')
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    plt.show()


def evaluate():
    """Train blood_model for a number of steps."""
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # randomize the inputs look
    x, y_, data, keep_prob = blood_model.prepare_input()
    # Get images and labels for blood_model.
    conv_output, W_conv1, W_conv2, h_conv1, h_conv2 = blood_model.inference(data, keep_prob)
    conv_predictions = blood_model.predictions(conv_output)

    sess = tf.InteractiveSession()

    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # extract global_step from it.
        global_step_number = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("checkpoint found at step %d", global_step_number)
    else:
        print('No checkpoint file found')
        return

    blood_dataset = np.load('../../labeller/data/wbc_p4-1_p.npy')
    blood_dataset = np.transpose(blood_dataset, (0, 2, 3, 1))
    predictions = sess.run(conv_predictions, feed_dict={x: blood_dataset, keep_prob: 1.0})
    np.save('../results/predictions.npy', np.argmax(predictions, 1))
    display_predictions(blood_dataset, np.argmax(predictions, 1))


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()