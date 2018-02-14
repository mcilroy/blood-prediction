import numpy as np
import tensorflow as tf
import os

DATA_LOCATION = '../../labeller/data/labelled_data'
FILE_LOCATION = 'pc9_with_vvc_7classes_training.npz'
VALIDATION_FILE_LOCATION = 'pc9_with_vvc_7classes_validation.npz'
USE_MULTIPLE_FILES = True
cell_names = ['neutrophils', 'monocytes', 'basophils', 'eosinophils', 'lymphocytes', 'strange_eosinophils', 'no_cells']
NUM_CLASSES = len(cell_names)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 11549


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


def inputs_balanced(batch_size, fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
    """
    create training, validation and testing datasets from data file
    """
    class DataSets(object):
            pass
    data_sets = DataSets()
    if fake_data:
        def fake():
            return DataSetBalanced([], [], batch_size, fake_data=True, one_hot=one_hot, dtype=dtype, eval_data=eval_data)
        data_sets.train = fake()
        data_sets.validation = fake()
        data_sets.test = fake()
        return data_sets

    #testing = dict()
    validation = dict()
    training = dict()
    validation_labels = dict()
    #testing_labels = dict()
    training_labels = dict()
    if USE_MULTIPLE_FILES:
        validation, validation_labels = create_data_set(VALIDATION_FILE_LOCATION, eval_data)
        if not eval_data:
            training, training_labels = create_data_set(FILE_LOCATION, eval_data)
            #### HACK: I needed to do this so there would be some strange eosinophil in the validation data ####
            validation['strange_eosinophils'] = training['strange_eosinophils'][0:10]
            validation_labels['strange_eosinophils'] = training_labels['strange_eosinophils'][0:10]
            training['strange_eosinophils'] = training['strange_eosinophils'][10:]
            training_labels['strange_eosinophils'] = training_labels['strange_eosinophils'][10:]
    else:
        VALIDATION_SIZE = 20
        #TESTING_SIZE = 1
        data_examples = np.load(os.path.join(DATA_LOCATION, FILE_LOCATION))
        for name in cell_names:
            print("data_examples")
            print(name+":"+str(data_examples[name].shape[0]))
        for i, name in enumerate(cell_names):
            if not eval_data:
                # make the random data consistent across runs
                np.random.seed(1)
                # Shuffle the data
                perm = np.arange(data_examples[name].shape[0])
                np.random.shuffle(perm)
                randomized_data = data_examples[name][perm]
            else:
                randomized_data = data_examples[name]
            validation[name] = randomized_data[:VALIDATION_SIZE]
            #testing[name] = randomized_data[VALIDATION_SIZE:VALIDATION_SIZE+TESTING_SIZE]
            if not eval_data:
                training[name] = randomized_data[VALIDATION_SIZE:]
                #training[name] = randomized_data[VALIDATION_SIZE+TESTING_SIZE:]
                training_labels[name] = to_categorical(np.full((training[name].shape[0], 1), i, dtype=int), NUM_CLASSES)
            validation_labels[name] = to_categorical(np.full((validation[name].shape[0], 1), i, dtype=int), NUM_CLASSES)
            #testing_labels[name] = to_categorical(np.full((testing[name].shape[0], 1), i, dtype=int), NUM_CLASSES)

    data_sets.validation = DataSetBalanced(validation, validation_labels, batch_size, fake_data=False, one_hot=True,
                                           dtype=tf.uint8, eval_data=eval_data)
    #data_sets.testing = DataSetBalanced(testing, testing_labels, batch_size, fake_data=False, one_hot=True, dtype=tf.uint8, eval_data=eval_data)
    if not eval_data:
        data_sets.train = DataSetBalanced(training, training_labels, batch_size, fake_data=False, one_hot=True,
                                          dtype=tf.uint8, eval_data=eval_data)

    return data_sets


def create_data_set(file_location, eval_data):
    """
    data_set is a dictionary which stores each cell type individually, used to create balanced datasets
    """
    data = np.load(os.path.join(DATA_LOCATION, file_location))
    print(file_location)
    for name in cell_names:
        print(name+":"+str(data[name].shape[0]))
    data_set = dict()
    data_set_labels = dict()
    for i, name in enumerate(cell_names):
        if not eval_data:
            # make the random data consistent across runs
            np.random.seed(1)
            # Shuffle the data
            perm = np.arange(data[name].shape[0])
            np.random.shuffle(perm)
            data_set[name] = data[name][perm]
        else:
            data_set[name] = data[name]
        data_set_labels[name] = to_categorical(np.full((data_set[name].shape[0], 1), i, dtype=int), NUM_CLASSES)
    return data_set, data_set_labels


class DataSetBalanced(object):
    def __init__(self, images, labels, batch_size, fake_data=False, one_hot=False, dtype=tf.float32, eval_data=False):
        """Construct a DataSet. one_hot arg is used only if fake_data is true.  `dtype` can be either `uint8` to leave
         the input as `[0, 255]`, or `float32` to rescale into `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            for name in cell_names:
                assert images[name].shape[0] == labels[name].shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
                print(name+":"+str(images[name].shape[0]))
            self._num_examples = 0
            for name in cell_names:
                self._num_examples += images[name].shape[0]
                assert images[name].shape[3] == 3
            if dtype == tf.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                for key in images:
                    images[key] = images[key].astype(np.float32)
                    images[key] = np.multiply(images[key], 1.0 / 255.0)
        self._epochs_completed = dict()
        self._index_in_epoch = dict()
        self._eval_data = eval_data
        for name in cell_names:
            self._epochs_completed[name] = 0
            self._index_in_epoch[name] = 0
            if not self._eval_data:
                # Shuffle the data
                perm = np.arange(images[name].shape[0])
                np.random.shuffle(perm)
                images[name] = images[name][perm]
                labels[name] = labels[name][perm]
        self._images = images
        self._labels = labels
        self.batch_size = batch_size
        if self._eval_data:
            self._images_original = dict()
            for name in cell_names:
                self._images_original[name] = np.copy(self._images[name])

    def get_batch(self, per_name_size):
        """Return the next 'per_name_size' examples from this data set.
        get indexes of data from each cell type (faster this way)
        when any cell type runs out, reshuffle that cell type"""
        start = dict()
        end = dict()
        for name in cell_names:
            start[name] = self._index_in_epoch[name]
            self._index_in_epoch[name] += per_name_size
            if self._index_in_epoch[name] > self._images[name].shape[0]:
                # Finished epoch
                self._epochs_completed[name] += 1
                # Shuffle the data
                perm = np.arange(self._images[name].shape[0])
                np.random.shuffle(perm)
                self._images[name] = self._images[name][perm]
                self._labels[name] = self._labels[name][perm]
                # Start next epoch
                start[name] = 0
                self._index_in_epoch[name] = per_name_size
                assert per_name_size <= self._images[name].shape[0]
            end[name] = self._index_in_epoch[name]
        return start, end

    def next_batch(self):
        """
        create batch: get images and labels from start and end indexes which are calcuated from 'get_batch'
        """
        assert self._eval_data is False
        per_name_size = self.batch_size/len(cell_names)
        start, end = self.get_batch(per_name_size)
        batch = np.empty((self.batch_size, self._images[cell_names[0]].shape[1],
                          self._images[cell_names[0]].shape[2], self._images[cell_names[0]].shape[3]))
        batch_labels = np.empty((self.batch_size, NUM_CLASSES))

        for i, name in enumerate(cell_names):
            batch[per_name_size*i:per_name_size*(i+1)] = self._images[name][start[name]:end[name]]
            batch_labels[per_name_size*i:per_name_size*(i+1)] = self._labels[name][start[name]:end[name]]
        return batch, batch_labels

    def next_batch_untouched(self):
        """
        create batch but also getting the original untouched images to use for displaying experimental results
        """
        assert self._eval_data is True
        per_name_size = self.batch_size/len(cell_names)
        start, end = self.get_batch(per_name_size)
        batch = np.empty((self.batch_size, self._images[cell_names[0]].shape[1],
                          self._images[cell_names[0]].shape[2], self._images[cell_names[0]].shape[3]))
        batch_labels = np.empty((self.batch_size, NUM_CLASSES))
        per_name_size = self.batch_size/len(cell_names)
        batch_original = np.empty((self.batch_size, self._images[cell_names[0]].shape[1],
                          self._images[cell_names[0]].shape[2], self._images[cell_names[0]].shape[3]))
        for i, name in enumerate(cell_names):
            batch[per_name_size*i:per_name_size*(i+1)] = self._images[name][start[name]:end[name]]
            batch_labels[per_name_size*i:per_name_size*(i+1)] = self._labels[name][start[name]:end[name]]
            batch_original[per_name_size*i:per_name_size*(i+1)] = self._images_original[name][start[name]:end[name]]
        return batch,  batch_original, batch_labels

    def get_all(self):
        """
        get all images, useful for testing large amounts of images at once
        """
        all = np.empty((self._num_examples, self._images[cell_names[0]].shape[1], self._images[cell_names[0]].shape[2],
                        self._images[cell_names[0]].shape[3]))
        all_labels = np.empty((self._num_examples, NUM_CLASSES))
        start = 0
        end = self._images[cell_names[0]].shape[0]
        #print("num examples "+str(self._num_examples))
        #print("all shape "+str(all.shape))
        for i, name in enumerate(cell_names):
            #print("start:"+str(start)+" end"+str(end))
            #print("images shape: "+str(self._images[name].shape))
            all[start:end] = self._images[name]
            all_labels[start:end] = self._labels[name]
            start = end
            if i+1 < len(cell_names):
                end += self._images[cell_names[i+1]].shape[0]
        return all, all_labels

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples

