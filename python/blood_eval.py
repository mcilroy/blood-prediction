import tensorflow as tf
import blood_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import blood_data
import os


FLAGS = tf.app.flags.FLAGS
RUN = 'pc9_with_vvc_7classes_tr35_va13'
tf.app.flags.DEFINE_string('checkpoint_dir', RUN+'/checkpoints', """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_string('batch_size', 90, """batch size""")
NUM_CLASSES = blood_data.NUM_CLASSES


def show_hard_images(images_used, batch_predictions):
    """
    display images which the network model had low confidence in 40-60%
    """
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
    """
    prints the confusion matrix between predictions and actual labels
                  actual
    #   predict 1 0 0 0 0
                0 1 0 0 0
    """
    pred = np.argmax(batch_predictions, 1)
    lab = np.argmax(labels, 1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for x in xrange(batch_predictions.shape[0]):
        matrix[pred[x], lab[x]] += 1
    #print(matrix)
    print('\n'.join([''.join(['{:4d}'.format(int(item)) for item in row]) for row in matrix]))
    correct_predictions = np.equal(pred, lab)
    tmp_accuracy = np.mean(correct_predictions)
    print("accuracy: " + str(tmp_accuracy))


def show_all_misclassified_images(images_used, batch_predictions, labels):
    """
    displays grid of misclassified predicted cell images visually. Organized to see what each cell's actual
     cell type is and which one it was predicted to be.
    """
    correct_predictions = np.equal(np.argmax(batch_predictions, 1), np.argmax(labels, 1))
    'actual: N, pred: M image1, image 2'
    'actual: N, pred: B image3'
    'actual: M, pred: N image4 image5'
    errors = [[[] for i in range(NUM_CLASSES)] for i in range(NUM_CLASSES)]
    for i, val in enumerate(correct_predictions):
        if not val:  # wrong
            ac = np.argmax(labels[i])
            pr = np.argmax(batch_predictions[i])
            errors[ac][pr].append(np.array(images_used[i], dtype='uint8'))

    #cell_names = ['neutrophils', 'monocytes', 'basophils', 'eosinophils', 'lymphocytes']
    cell_names = ['Ne', 'Mo', 'Ba', 'Eo', 'Ly', 'str eosin', 'no cell']
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


def plot_coulter_vs_predictions(predictions, labels):
    """ Compare the coulter counter counts of cells in the image with the predicted counts based on the
     individual predictions of cell images.
     Plot the arrays of data against each other. They should plot a straight line.
     Use polynomial regression to evaluate how well they match.
    """
    # load coulter counter data
    NEUTROPHIL = "neutrophil"
    MONOCYTE = "monocyte"
    LYMPHOCYTE = "lymphocyte"
    BASOPHIL = "basophil"
    EOSINOPHIL = "eosinophil"
    coulter_labels = dict()
    counts = np.load(os.path.join("../../labeller/data", "coulter_count", 'V_vs_C_cc_scaled.npz'))
    coulter_labels['neutrophils'] = counts[NEUTROPHIL]
    coulter_labels['monocytes'] = counts[MONOCYTE]
    coulter_labels['basophils'] = counts[BASOPHIL]
    coulter_labels['eosinophils'] = counts[EOSINOPHIL]
    coulter_labels['lymphocytes'] = counts[LYMPHOCYTE]
    coulter_labels['wbc'] = counts['wbc']

    cell_names_predictions = ['neutrophils', 'monocytes', 'basophils', 'eosinophils', 'lymphocytes', 'strange_eosinophils', 'no_cells']
    patient_data = np.load(os.path.join("../../labeller/data/labelled_data", "pc9_with_vvc_7classes_validation_patients.npz"))
    patient_data_all = np.concatenate((patient_data['neutro_patients'], patient_data['mono_patients'], patient_data['baso_patients'],
                    patient_data['eosin_patients'], patient_data['lymp_patients'],
                    patient_data['strange_eosin_patients'], patient_data['no_cell_patients']))
    num_patients = len(np.unique(patient_data_all))
    patient_counts = dict()

    for cell_name in cell_names_predictions:
        patient_counts[cell_name] = np.zeros(num_patients)
    patient_counts['wbc'] = np.zeros(num_patients)
    pred_col = np.argmax(predictions, 1)
    for i, class_idx in enumerate(pred_col):
        patient_idx = int(patient_data_all[i])
        if cell_names_predictions[class_idx] == 'no_cells':
            continue
        if cell_names_predictions[class_idx] == 'strange_eosinophils':
            patient_counts['neutrophils'][patient_idx] += 1
        else:
            patient_counts[cell_names_predictions[class_idx]][patient_idx] += 1
        patient_counts['wbc'][patient_idx] += 1
    del patient_counts['strange_eosinophils']
    del patient_counts['no_cells']

    # remove NaNs
    nan_indexes = dict()
    for key in coulter_labels:
        mask = np.ones(len(coulter_labels[key]), dtype=bool)
        nan_indexes[key] = [i for i, v in enumerate(coulter_labels[key]) if np.isnan(v)]
        mask[nan_indexes[key]] = False
        coulter_labels[key] = coulter_labels[key][mask]
    indexes_remaining = dict()
    for key in patient_counts:
        mask = np.ones(len(patient_counts[key]), dtype=bool)
        mask[nan_indexes[key]] = False
        indexes_remaining[key] = []
        for i, v in enumerate(mask):
            if v:
                indexes_remaining[key].append(i)
        patient_counts[key] = patient_counts[key][mask]

    # plot data
    fig = plt.figure(5)
    count = 1
    print_string = ""
    for key, value in coulter_labels.iteritems():
        rval = polyfit(coulter_labels[key], patient_counts[key], 1)
        print_string += key + " " + str(rval['determination']).format('%f')+"\n"
        #ax1 = self.fig.add_subplot(111)
        plt.subplot(3, 2, count)
        #self.fig.add_subplot(blah)
        count += 1
        length = max(max(patient_counts[key]), max(coulter_labels[key]))
        plt.axis([0, length, 0, length])
        plt.scatter(patient_counts[key], coulter_labels[key])
        plt.title(key)
        plt.ylabel('Coulter Counter')
        plt.xlabel('Predictions')
    print(print_string)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
    #self.fig.tight_layout()
    ######plt.show()
    fig.savefig("../results/predictions_vs_coulter_counter", dpi=100)


def polyfit(x, y, degree):
    """ Polynomial Regression """
    results = {}
    coeffs = np.polyfit(x, y, degree)
     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot
    return results


def get_prediction_labels(cell_names, predictions, labels):
    """ determine the number of correct predictions """
    prediction_labels = dict()
    correct_predictions = np.equal(np.argmax(predictions, 1), np.argmax(labels, 1))
    for i, cor_pred in enumerate(correct_predictions):
        if cor_pred:
            prediction_labels[cell_names[np.argmax(predictions[i], 1)]] += 1


def show_misclassified_images(images_used, batch_predictions, labels):
    """ Show first 4 misclassified images (for each cell type) in a grid """
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
    """ display filters in a grid from first layer """
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
    """ display filters of 1st layer """
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
    """Load a saved model and generate predictions on test data. Then do experiments to evaluate
    predictions."""
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

    # batch_val = blood_datasets.validation.next_batch_untouched()  # [[70, 81, 81, 3], [70, 81, 81, 3], [70, 7]]
    # predictions = sess.run(conv_predictions, feed_dict={x: batch_val[0], y_: batch_val[2], keep_prob: 1.0})  # [70, 7]
    # print_confusion_matrix(predictions, batch_val[2])
    # show_all_misclassified_images(batch_val[1], predictions, batch_val[2])

    predictions = np.empty((blood_datasets.validation.num_examples, 7))
    labels = np.empty((blood_datasets.validation.num_examples, 7))
    images = np.empty((blood_datasets.validation.num_examples, 81, 81, 3))
    #images_untouched = np.empty((blood_datasets.validation.num_examples, 81, 81, 3))

    all_data, all_labels = blood_datasets.validation.get_all()
    for i in xrange(int(blood_datasets.validation.num_examples / FLAGS.batch_size)):
        batch_val = [all_data[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size, :, :, :], all_labels[i*FLAGS.batch_size: (i+1)*FLAGS.batch_size, :]]
        temp_predictions = sess.run(conv_predictions, feed_dict={x: batch_val[0], y_: batch_val[1], keep_prob: 1.0})
        predictions[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :] = temp_predictions
        images[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :] = batch_val[0]
        #images_untouched[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :] = batch_val[1]
        labels[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size, :] = batch_val[1]
    print_confusion_matrix(predictions, labels)
    plot_coulter_vs_predictions(predictions, labels)
    #show_all_misclassified_images(images, predictions, labels)


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
