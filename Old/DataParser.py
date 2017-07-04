import random
from matplotlib import pyplot as plt
from six.moves import cPickle as pickle
import os
import numpy as np
from scipy.misc import imresize

# Parses the data, spliting the images into single digits, resizing and making a train and vaidation set

pixel_depth = 255.0
aspect_ratio = [32, 32]


def split_image_into_single_digits(image, metadata):
    labels = list()
    digits = list()

    for index in range(len(metadata['label'])):
        # get metadata
        height = int(metadata['height'][index])
        left = int(metadata['left'][index])
        top = int(metadata['top'][index])
        width = int(metadata['width'][index])
        # expand boxes for a couple more pixels
        alpha = int(.1 * max(height, width))
        height += 2 * alpha
        left -= alpha
        top -= alpha
        width += 2 * alpha

        # fix negative values for left and top
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        # get digit
        digit = image[top:(top + height), left:(left + width)]
        # pad digit
        m = max(height, width)
        vertical_pad = (m - height) / 2
        horizontal_pad = (m - width) / 2
        digit = np.lib.pad(digit, ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)), 'mean')
        # resize to aspect ratio, scale, and append digit and label
        im = imresize(digit, aspect_ratio).astype(float)
        im = (im - pixel_depth / 2) / pixel_depth
        digits.append(im)
        labels.append(metadata['label'][index])

    return digits, labels


def turn_dataset_into_non_sequential(dataset):
    labels = list()
    digits = list()

    for key in dataset:
        image = dataset[key][0]
        metadata = dataset[key][1]
        d, l = split_image_into_single_digits(image, metadata)
        digits += d
        labels += l

    return np.array(digits, dtype=np.float32), np.array(labels)


def visualize_examples(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        train_set = data['train_set']

        indices = random.sample(range(len(train_set)), 5)
        for index in indices:
            plt.imshow(train_set[index], interpolation='nearest')
            plt.show()


def build_sets(filename, valid_set_percentage, force=False):
    if os.path.exists(filename) and not force:
        print('%s already present - Skipping pickling.' % filename)
        return

    train = pickle.load(open('train.pickle', 'rb'))

    set_, labels = turn_dataset_into_non_sequential(train)

    split = int(round(len(labels) * (1 - valid_set_percentage), 0))
    end = len(labels)
    train_set = set_[0:split]
    train_labels = labels[0:split]
    valid_set = set_[split + 1:end]
    valid_labels = labels[split + 1:end]

    test = pickle.load(open('test.pickle', 'rb'))
    test_set, test_labels = turn_dataset_into_non_sequential(test)

    print('Pickling %s.' % filename)
    try:
        with open(filename, 'wb') as f:
            data = {'train_set': train_set, 'train_labels': train_labels, 'valid_set': valid_set,
                    'valid_labels': valid_labels, 'test_set': test_set, 'test_labels': test_labels,
                    'aspect_ratio': aspect_ratio}
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', filename, ':', e)
    return


filename = 'sets.pickle'
build_sets(filename, .1, force=True)
visualize_examples(filename)
