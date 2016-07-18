import tensorflow as tf
import blood_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

FLAGS = tf.app.flags.FLAGS
RUN = 'all_five_cells_balanced_paul_sameseed'
tf.app.flags.DEFINE_string('checkpoint_dir', RUN+'/checkpoints', """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_string('batch_size', 90, """batch size""")


def show_hard_images(images_used, batch_predictions):

    fig = plt.figure()
    fig.suptitle('model is unsure 40-60% confidence', fontsize=14, fontweight='bold')
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


def print_confusion_matrix(batch_predictions, labels):
    """           actual
    #   predict 1 0 0 0 0
                0 1 0 0 0
    """
    pred = np.argmax(batch_predictions, 1)
    lab = np.argmax(labels, 1)
    matrix = np.zeros((5, 5))
    for x in xrange(batch_predictions.shape[0]):
        matrix[pred[x], lab[x]] += 1
    print(matrix)
    correct_predictions = np.equal(pred, lab)
    tmp_accuracy = np.mean(correct_predictions)
    print("accuracy: " + str(tmp_accuracy))


def show_all_misclassified_images(images_used, batch_predictions, labels):
    correct_predictions = np.equal(np.argmax(batch_predictions, 1), np.argmax(labels, 1))
    'actual: N, pred: M image1, image 2'
    'actual: N, pred: B image3'
    'actual: M, pred: N image4 image5'
    errors = [[[] for i in range(5)] for i in range(5)]
    for i, val in enumerate(correct_predictions):
        if not val:  # wrong
            ac = np.argmax(labels[i])
            pr = np.argmax(batch_predictions[i])
            errors[ac][pr].append(np.array(images_used[i], dtype='uint8'))

    #cell_names = ['neutrophils', 'monocytes', 'basophils', 'eosinophils', 'lymphocytes']
    cell_names = ['Ne', 'Mo', 'Ba', 'Eo', 'Ly']
    cols = []
    cols_images = []
    row_lens = []
    for i, actual in enumerate(errors):
        for j, pred in enumerate(actual):
            if pred != []:
                cols.append('Actual: '+cell_names[i]+", Pred: "+cell_names[j])
                cols_images.append(pred)
                row_lens.append(len(errors[i][j]))
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(max(row_lens), len(cols)), axes_pad=0.1,)

    blank = np.zeros((75, 75, 3))
    for x in xrange(max(row_lens)*len(cols)):
        grid[x].imshow(blank)
    for i, col in enumerate(cols):
        for j, image in enumerate(cols_images[i]):
            grid[(len(cols)*j)+i].imshow(image)

    pad = 5  # in points
    for ax, col in zip(grid.axes_all, cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points', size='large',
                    ha='center', va='baseline')
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    plt.show()


def show_misclassified_images(images_used, batch_predictions, labels):
    correct_predictions = np.equal(np.argmax(batch_predictions, 1), np.argmax(labels, 1))

    fig = plt.figure()
    fig.suptitle('neutrophil, monocyte, basophil, eosinophil, lymphocyte', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5),
                     axes_pad=0.1,)
    count_neutro = 0
    count_mono = 0
    count_baso = 0
    count_eosin = 0
    count_lymph = 0
    blank = np.zeros((75, 75, 3))
    for x in xrange(5*5):
        grid[x].imshow(blank)
    for i, val in enumerate(correct_predictions):
        if not val:  # wrong
            image = np.array(images_used[i], dtype='uint8')
            if labels[i, 0] == 1:  # neutrophile
                if count_neutro < 5:
                    grid[count_neutro*5].imshow(image)
                    count_neutro += 1
            elif labels[i, 1] == 1:  # mono
                if count_mono < 5:
                    grid[(count_mono*5)+1].imshow(image)
                    count_mono += 1
            elif labels[i, 2] == 1:  # basophil
                if count_baso < 5:
                    grid[(count_baso*5)+2].imshow(image)
                    count_baso += 1
            elif labels[i, 3] == 1:  # eosin
                if count_eosin < 5:
                    grid[(count_eosin*5)+3].imshow(image)
                    count_eosin += 1
            elif labels[i, 4] == 1:  # lymph
                if count_lymph < 5:
                    grid[(count_lymph*5)+4].imshow(image)
                    count_lymph += 1
        if count_neutro >= 4 and count_mono >= 4 and count_baso >= 4 and count_eosin >= 4 and count_lymph >= 4:
            break
    #print("mislabelled image count " + str(count_mono+count_neutro))
    plt.show()


def show_filters(filters):
    filters = np.rollaxis(filters, 3, 0)
    fig = plt.figure()
    fig.suptitle('filters 1st layer', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=(filters.shape[0]/4, 3),
                     axes_pad=0.1,)
    for i in range(filters.shape[0]/4):
        for j in range(3):
            single_filter = filters[i, :, :, j]  # 5x5
            blah = np.zeros([single_filter.shape[0], single_filter.shape[1], 3])#5x5x3
            blah[:, :, j] = single_filter
            grid[(i*3)+j].imshow(blah)
    print("filter # : " + str(filters.shape[0]))
    plt.show()


def show_filters_alt(filters):
    filters = np.rollaxis(filters, 3, 0)
    fig = plt.figure()
    fig.suptitle('filters 1st layer', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=(filters.shape[0]/4, 2),
                     axes_pad=0.1,)
    for i in range(filters.shape[0]/4):
        for j in range(2):
            single_filter = filters[i, :, :, :]  # 5x5x3
            grid[(i*2)+j].imshow(single_filter)
    print("filter # : " + str(filters.shape[0]))
    plt.show()


def show_image_response_to_filters(activations, images):
    """find the images that respond maximally to each filter
    activation_1: 50x81x81x32 -> 50x(81*81)x32 -> sum to 50x1x32 -> select for each filter the index with the highest
    value, display the image
    activation_2: 50x41x41x64 """
    activations_reshape = np.reshape(activations, [activations.shape[0], activations.shape[1]*activations.shape[2],
                                                   activations.shape[3]])
    activations_sum = np.sum(activations_reshape, axis=1)
    top = 3
    filter_amount = activations.shape[3]/8
    fig = plt.figure()
    fig.suptitle('filters 1st layer', fontsize=14, fontweight='bold')
    grid = ImageGrid(fig, 111, nrows_ncols=(filter_amount, top),
                     axes_pad=0.1,)

    for x in range(filter_amount):  # 32
        #max_index = np.argmax(activations_sum[:, 0, x])
        arr = activations_sum[:, x]
        indexes_top = arr.argsort()[-top:][::-1]
        for i, idx in enumerate(indexes_top):  # 3
            grid[(x*top)+i].imshow(images[idx, :, :, :])
    plt.show()


def eval_once():
    pass


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
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("checkpoint found at step %d", global_step)
    else:
        print('No checkpoint file found')
        return

    blood_datasets = blood_model.inputs(eval_data=True)

    batch_val = blood_datasets.validation.next_batch_untouched()
    predictions = sess.run(conv_predictions, feed_dict={x: batch_val[0], y_: batch_val[2], keep_prob: 1.0})
    print_confusion_matrix(predictions, batch_val[2])
    show_all_misclassified_images(batch_val[1], predictions, batch_val[2])

    #predictions = sess.run(conv_predictions, feed_dict={x: blood_datasets.validation.images, y_: blood_datasets.validation.labels, keep_prob: 1.0})
    #print_confusion_matrix(predictions, blood_datasets.validation.labels)
    #show_misclassified_images(blood_datasets.validation._images_original, predictions, blood_datasets.validation.labels)

    # show_hard_images(batch_val[1], predictions)
    # filters = sess.run(W_conv1)
    # show_filters(filters)
    # show_filters_alt(filters)
    # activations_2 = sess.run(h_conv2, feed_dict={x: batch_val[0], y_: batch_val[2], keep_prob: 1.0})
    # show_image_response_to_filters(activations_2, batch_val[1])
    # activations_2 = sess.run(h_conv2, feed_dict={x: blood_datasets.validation.images, y_: blood_datasets.validation.labels, keep_prob: 1.0})
    # show_image_response_to_filters(activations_2, blood_datasets.validation.images)


def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()
