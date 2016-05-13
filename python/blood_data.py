import numpy as np
import tensorflow as tf
import os

DATA_LOCATION = '../../../AlanFine'


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


def inputs(fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
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
    def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
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
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._eval_data = eval_data
        if self._eval_data:
            self._images_original = np.copy(self._images)

        self.mean = np.mean(self._images)
        #self.std = max(np.std(self._images), 1.0/np.sqrt(self._images.size))
        self.var = np.var(self._images)
        self._images = (self._images - self.mean)/self.var

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 81*81
            if self.one_hot:
                fake_label = [1] + [0] * 2
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def next_batch_untouched(self, batch_size, fake_data=False):
        """Return the next "batch_size" examples from this data set."""
        if fake_data:
            fake_image = [1] * 81*81
            if self.one_hot:
                fake_label = [1] + [0] * 2
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        assert self._eval_data is True
        return self._images[start:end], self._images_original[start:end], self._labels[start:end]

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
