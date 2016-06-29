import numpy as np
import tensorflow as tf
import os

DATA_LOCATION = '../../labeller/data'


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    for i, val in enumerate(labels_dense):
        if val == 1:
            labels_one_hot[i, 0], labels_one_hot[i, 1] = 0, 1
        else:
            labels_one_hot[i, 0], labels_one_hot[i, 1] = 1, 0
    return labels_one_hot


def to_categorical(y, nb_classes):
    """ to_categorical.
    Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    Arguments:
        y: `array`. Class vector to convert.
        nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def distort(examples):
    x = tf.placeholder(tf.float32, shape=[81, 81, 3])

    distorted_image = tf.random_crop(x, [75, 75, 3])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    #distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    with tf.Session() as sess:
        distorted_images = np.zeros((examples.shape[0]*10, 75, 75, examples.shape[3]))
        for i in xrange(len(examples)):
            image = examples[i, :, :, :]
            for j in xrange(10):
                reshaped_image = sess.run([float_image], feed_dict={x: image})
                numpyarray = np.array(reshaped_image[0])  # list of [1, 75, 75, 3]
                distorted_images[(i*10)+j, :, :, :] = numpyarray
    return distorted_images


def undistorted(examples, whitening):
    x = tf.placeholder(tf.float32, shape=[81, 81, 3])

    undistorted_image = tf.random_crop(x, [75, 75, 3])
    if whitening:
        # Subtract off the mean and divide by the variance of the pixels.
        undistorted_float_image = tf.image.per_image_whitening(undistorted_image)
    else:
        undistorted_float_image = undistorted_image
    with tf.Session() as sess:
        if whitening:
            undistorted_images = np.zeros((examples.shape[0], 75, 75, examples.shape[3]))
        else:
            undistorted_images = np.zeros((examples.shape[0], 75, 75, examples.shape[3]), dtype="uint8")
        for i in xrange(len(examples)):
            image = examples[i, :, :, :]
            reshaped_image = sess.run([undistorted_float_image], feed_dict={x: image})
            numpyarray = np.array(reshaped_image[0])  # list of [1, 75, 75, 3]
            undistorted_images[i, :, :, :] = numpyarray
    return undistorted_images


def concatenate_data(data_examples, method, start, end, length, whitening, training):
    if training:
        neutrophils = method(data_examples['neutrophils'][start:end])
        monocytes = method(data_examples['monocytes'][start:end])
        basophils = method(data_examples['basophils'][start:end])
        eosinophils = method(data_examples['eosinophils'][start:end])
        lymphocytes = method(data_examples['lymphocytes'][start:end])
    else:
        neutrophils = method(data_examples['neutrophils'][start:end], whitening)
        monocytes = method(data_examples['monocytes'][start:end], whitening)
        basophils = method(data_examples['basophils'][start:end], whitening)
        eosinophils = method(data_examples['eosinophils'][start:end], whitening)
        lymphocytes = method(data_examples['lymphocytes'][start:end], whitening)
    validation_images = np.concatenate((neutrophils, monocytes, basophils, eosinophils, lymphocytes))
    if training:
        validation_labels = np.concatenate((np.full((neutrophils, 1), 0, dtype=int),
                                        np.full((monocytes, 1), 1, dtype=int),
                                        np.full((basophils, 1), 2, dtype=int),
                                        np.full((eosinophils, 1), 3, dtype=int),
                                        np.full((lymphocytes, 1), 4, dtype=int)))
    else:
        validation_labels = np.concatenate((np.full((length, 1), 0, dtype=int),
                                        np.full((length, 1), 1, dtype=int),
                                        np.full((length, 1), 2, dtype=int),
                                        np.full((length, 1), 3, dtype=int),
                                        np.full((length, 1), 4, dtype=int)))
    validation_labels = to_categorical(validation_labels, 5)
    return validation_images, validation_labels


def inputs(fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
    """
    take x amount from un-distorted data for testing and validation
    distort the rest of the data for training
    :param fake_data:
    :param one_hot:
    :param dtype:
    :param eval_data:
    :return: distorted images
    """
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    VALIDATION_SIZE = 5
    TESTING_SIZE = 2
    data_examples = np.load(os.path.join(DATA_LOCATION, 'pc9_collages_cleaned.npz'))

    # if not eval:
    #   make val, test with images undistorted method (with whitening)
    #   train with distorted method
    #   skip doing val, test originals with undistorted method (without whitening)
    #   leave original data in dataset method empty
    # else if eval:
    #   make val, test with images undistorted method (with whitening)
    #   skip train with distorted method
    #   do val, test originals with undistorted method (without whitening)
    #   pass original data in dataset method

    if eval:
        validation_images, validation_labels = concatenate_data(data_examples=data_examples, method=undistorted,
                                                                start=0, end=VALIDATION_SIZE, length=VALIDATION_SIZE,
                                                                whitening=True, training=False)
        testing_images, testing_labels = concatenate_data(data_examples=data_examples, method=undistorted,
                                                          start=VALIDATION_SIZE, end=VALIDATION_SIZE+TESTING_SIZE,
                                                          length=TESTING_SIZE, whitening=True, training=False)
        validation_images_original, validation_labels_original = concatenate_data(data_examples=data_examples, method=undistorted,
                                                                start=0, end=VALIDATION_SIZE, length=VALIDATION_SIZE,
                                                                whitening=False, training=False)
        testing_images_original, testing_labels_original = concatenate_data(data_examples=data_examples, method=undistorted,
                                                          start=VALIDATION_SIZE, end=VALIDATION_SIZE+TESTING_SIZE,
                                                          length=TESTING_SIZE, whitening=False, training=False)

        data_sets.validation = DataSet(validation_images, validation_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                   eval_data=eval_data, original_images=validation_images_original)
        data_sets.testing = DataSet(testing_images, testing_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                eval_data=eval_data, original_images=testing_images_original)
    else:
        validation_images, validation_labels = concatenate_data(data_examples=data_examples, method=undistorted,
                                                                start=0, end=VALIDATION_SIZE, length=VALIDATION_SIZE,
                                                                whitening=True, training=False)
        testing_images, testing_labels = concatenate_data(data_examples=data_examples, method=undistorted,
                                                          start=VALIDATION_SIZE, end=VALIDATION_SIZE+TESTING_SIZE,
                                                          length=TESTING_SIZE, whitening=True, training=False)
        training_images, training_labels = concatenate_data(data_examples=data_examples, method=distort,
                                                          start=VALIDATION_SIZE+TESTING_SIZE, end=None,
                                                          length=TESTING_SIZE, whitening=True, training=True)

        data_sets.validation = DataSet(validation_images, validation_labels, fake_data=False, one_hot=True,
                                       dtype=tf.uint8, eval_data=eval_data)
        data_sets.testing = DataSet(testing_images, testing_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                    eval_data=eval_data)
        data_sets.training = DataSet(training_images, training_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                     eval_data=eval_data)
    return data_sets


def inputs_neutro_mono_only(fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets
    TRAINING_SIZE = 1000
    VALIDATION_SIZE = 294
    train_val = np.load(os.path.join(DATA_LOCATION, 'monocytes_neutrophils.npz'))
    test = np.load(os.path.join(DATA_LOCATION, 'mono_neut_test.npz'))
    # stored as batch, depth, height, width. Tensorflow wants -> batch, height, width, depth
    neutrophils = np.rollaxis(train_val['neutrophils'], 1, 4)
    monocytes = np.rollaxis(train_val['monocytes'], 1, 4)

    test_neutrophils = np.rollaxis(test['neut'], 1, 4)
    test_monocytes = np.rollaxis(test['mono'], 1, 4)

    validation_images = np.concatenate([neutrophils[:VALIDATION_SIZE],
                                        monocytes[:VALIDATION_SIZE]])
    validation_labels = np.concatenate([
        dense_to_one_hot(np.ones((VALIDATION_SIZE, 1), dtype=np.int), 2),
        dense_to_one_hot(np.zeros((VALIDATION_SIZE, 1), dtype=np.int), 2)])

    training_images = np.concatenate([neutrophils[VALIDATION_SIZE:], monocytes[VALIDATION_SIZE:]])
    training_labels = np.concatenate([
        dense_to_one_hot(np.ones((neutrophils.shape[0]-VALIDATION_SIZE, 1), dtype=np.int), 2),
        dense_to_one_hot(np.zeros((monocytes.shape[0] - VALIDATION_SIZE, 1), dtype=np.int), 2)])

    testing_images = np.concatenate([test_neutrophils, test_monocytes])
    testing_labels = np.concatenate([dense_to_one_hot(np.ones((test_neutrophils.shape[0], 1), dtype=np.int), 2),
                                     dense_to_one_hot(np.zeros((test_monocytes.shape[0], 1), dtype=np.int), 2)])

    data_sets.train = DataSet(training_images, training_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                              eval_data=eval_data)
    data_sets.validation = DataSet(validation_images, validation_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                   eval_data=eval_data)
    data_sets.testing = DataSet(testing_images, testing_labels, fake_data=False, one_hot=True, dtype=tf.uint8,
                                eval_data=eval_data)
    return data_sets


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False, original_images=None):
        """Construct a DataSet. one_hot arg is used only if fake_data is true.  `dtype` can be either `uint8` to leave
         the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`. """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            assert images.shape[3] == 3
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            # images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            if dtype == tf.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(np.float32)
                images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.num_examples = len(self._images)
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._eval_data = eval_data
        if self._eval_data:
            self._images_original = original_images  #np.copy(self._images)
        #self._images = undistorted(self._images, False)
        #self._images_original = undistorted(self._images_original, True)

    def get_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self._index_in_epoch
        return start, end

    def next_batch(self, batch_size):
        assert self._eval_data is False
        start, end = self.get_batch(batch_size)
        return self._images[start:end], self._labels[start:end]

    def next_batch_untouched(self, batch_size):
        assert self._eval_data is True
        start, end = self.get_batch(batch_size)
        return self._images[start:end],  self._images_original[start:end], self._labels[start:end]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
