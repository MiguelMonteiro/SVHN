from __future__ import print_function
import numpy as np
import os
import sys
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import scipy.io
import random
from ImagePreProcessingUtils import im2gray, visualize_some_examples, global_contrast_normalization

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None


def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    return filename


def fix_data_shape(data):
    return data.transpose(3, 0, 1, 2)


def fix_labels(labels):
    labels[labels == 10] = 0
    return labels.squeeze()


def shuffle_data(data, labels):
    order = range(len(labels))
    random.shuffle(order)
    return data[order], labels[order]


def split_into_train_and_validation_set(train_data, train_labels, valid_percentage=.1):
    split = int((1 - valid_percentage) * len(train_labels))
    return train_data[:split], train_labels[:split], train_data[split:], train_labels[split:]


def pre_process_images(data):
    return global_contrast_normalization(im2gray(data)).astype(np.float32)




train_matfile = maybe_download('data/train_32x32.mat')
test_matfile = maybe_download('data/test_32x32.mat')
extra_matfile = maybe_download('data/extra_32x32.mat')


train_data = scipy.io.loadmat('data/train_32x32.mat', variable_names='X').get('X')
train_labels = scipy.io.loadmat('data/train_32x32.mat', variable_names='y').get('y')
test_data = scipy.io.loadmat('data/test_32x32.mat', variable_names='X').get('X')
test_labels = scipy.io.loadmat('data/test_32x32.mat', variable_names='y').get('y')
#extra_data = scipy.io.loadmat('data/extra_32x32.mat', variable_names='X').get('X')
#extra_labels = scipy.io.loadmat('data/extra_32x32.mat', variable_names='y').get('y')

train_data = fix_data_shape(train_data)
test_data = fix_data_shape(test_data)
#extra_data = fix_data_shape(extra_data)

train_labels = fix_labels(train_labels)
test_labels = fix_labels(test_labels)
#extra_labels = fix_labels(extra_labels)

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
#print(extra_data.shape, extra_labels.shape)

#all_train_data = np.concatenate((train_data, extra_data), axis=0)
#all_train_labels = np.concatenate((train_labels, extra_labels), axis=0)
all_train_data = train_data
all_train_labels = train_labels

train_data, train_labels = shuffle_data(train_data, train_labels)
train_data, train_labels, valid_data, valid_labels = split_into_train_and_validation_set(train_data, train_labels)


train_data = pre_process_images(train_data)
valid_data = pre_process_images(valid_data)
test_data = pre_process_images(test_data)

visualize_some_examples(train_data, train_labels)

filename = 'data/SVHN.pickle'
with open(filename, 'wb') as f:
    print('Pickling file: {0}'.format(filename))
    save = {'train_set': train_data, 'train_labels': train_labels,
            'valid_set': valid_data, 'valid_labels': valid_labels,
            'test_set': test_data, 'test_labels': test_labels}
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    statinfo = os.stat(filename)
    print('Compressed pickle size:', statinfo.st_size/1e6, ' MB')