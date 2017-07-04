# Loads the data from the train and test folders, images and metadata, and puts it into a more manageble format

import os
from scipy import ndimage
from six.moves import cPickle as pickle
import h5py


def load_image(folder):
    image_files = os.listdir(folder)
    dataset = dict()

    n_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        if(n_images%1000 == 0):
            print (n_images)
        try:
            #flatten = true makes image grayscale
            image_data = ndimage.imread(image_file, flatten=True).astype(float)
            dataset[image] = image_data
            n_images += 1
        except:
            print('Could not read:', image_file, ': - it\'s ok, skipping.')
    return dataset


def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_meta_data(folder):
    path = folder + '/digitStruct.mat'
    f = h5py.File(path,'r')
    name_pointers = f['digitStruct/name']

    metadata = dict()
    for index in range(len(name_pointers)):
        metadata[get_name(index, f)] = get_box_data(index, f)
    return metadata


def maybe_pickle(folder, force=False):
    set_filename = folder + '.pickle'
    if os.path.exists(set_filename) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping pickling.' % set_filename)
    else:
        print('Pickling %s.' % set_filename)
        images = load_image(folder)
        metadata = get_meta_data(folder)
        dataset = {k: [images[k], metadata[k]] for k in images}
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)
    return set_filename

train_folder = "train"
train_dataset = maybe_pickle(train_folder, len(os.listdir(train_folder)))
test_folder = "test"
test_dataset = maybe_pickle(test_folder, len(os.listdir(test_folder)))