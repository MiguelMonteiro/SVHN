from __future__ import print_function
import os
import sys
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import scipy.io
import random
import numpy as np

# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#
#
# def save_to_tf_record(data, labels, file_name):
#     tfrecords_filename = file_name + '.tfrecords'
#
#     writer = tf.python_io.TFRecordWriter(tfrecords_filename)
#
#     for im, l in zip(data, labels):
#         np.set_printoptions(precision=128)
#         shape = np.array(im.shape).tostring()
#         img_raw = im.tostring()
#         l_raw = l.tostring()
#
#         example = tf.train.Example(features=tf.train.Features(feature={
#             'shape': _bytes_feature(shape),
#             'image_raw': _bytes_feature(img_raw),
#             'label_raw': _bytes_feature(l_raw)}))
#
#         writer.write(example.SerializeToString())
#     writer.close()


def im2gray(image):
    return np.dot(image, [[0.2989], [0.5870], [0.1140]])


# this form of normalization gives very poor results
def contrast_normalization(image, pixel_depth=255.0):
    return (image - pixel_depth / 2) / pixel_depth


# adjusts the contrast of each image individually, performs much better than the method above
def global_contrast_normalization(images, min_divisor=1e-4):
    num_images = images.shape[0]
    mean = np.mean(images, axis=(1, 2), dtype=float)
    std = np.std(images, axis=(1, 2), dtype=float, ddof=1)
    std[std < min_divisor] = 1.

    images_gcn = np.zeros(images.shape, dtype=float)
    for i in np.arange(num_images):
        images_gcn[i, :, :] = (images[i, :, :] - mean[i]) / std[i]

    return images_gcn


def visualize_some_examples(data, labels):
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    f, ax = plt.subplots(nrows=1, ncols=10)

    for i, j in enumerate(np.random.randint(0, len(labels), size=10)):
        ax[i].axis('off')
        ax[i].set_title(labels[j], loc='center')
        ax[i].imshow(data[j].squeeze())
    plt.show()


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

#visualize_some_examples(train_data, train_labels)

save_to_tf_record(train_data, train_labels,'asdf')

# filename = 'data/SVHN.pickle'
# with open(filename, 'wb') as f:
#     print('Pickling file: {0}'.format(filename))
#     save = {'train_set': train_data, 'train_labels': train_labels,
#             'valid_set': valid_data, 'valid_labels': valid_labels,
#             'test_set': test_data, 'test_labels': test_labels}
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
#     statinfo = os.stat(filename)
#     print('Compressed pickle size:', statinfo.st_size/1e6, ' MB')