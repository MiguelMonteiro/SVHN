import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.python.lib.io import file_io

from SingleDigitClassification.old.CNN import ConventionalCNN


def load_data(file_path):
    with file_io.FileIO(file_path, 'r') as f:
        data = pickle.load(f)
        train_set, train_labels, valid_set, valid_labels, test_set, test_labels = \
            [data[key] for key in ['train_set', 'train_labels', 'valid_set', 'valid_labels', 'test_set', 'test_labels']]

    return train_set, train_labels, valid_set, valid_labels, test_set, test_labels


def run_experiment(data_path, output_path):

    train_set, train_labels, valid_set, valid_labels, test_set, test_labels = load_data(data_path)

    # this is the Udacity's course architecture
    batch_size = 64
    image_size = np.array([32, 32])
    num_channels = 1
    num_labels = 10

    # 3 convolutional layers with the same patch size and stride
    depths = [16, 32, 64]
    n = len(depths)
    patch_size = [np.array([5, 5])] * n
    conv_strides = [np.array([1, 1])] * n
    padding = ['VALID'] * n
    # 2 max pooling layers between the convolutional layers
    pooling_strides = [np.array([2, 2])] * (n - 1)
    pooling_filter_sizes = [np.array([2, 2])] * (n - 1)
    # one hidden layer (not including the connecting hidden layer whose size is defined by the convolutional layers)
    num_hidden = [32]

    net = ConventionalCNN(image_size, batch_size, num_channels, num_labels, patch_size, depths, conv_strides,
                          pooling_filter_sizes, pooling_strides, num_hidden)
    graph = net.get_graph()

    prev_valid_accuracy = 0

    train_writer = tf.summary.FileWriter(output_path + '/train', graph)
    valid_writer = tf.summary.FileWriter(output_path + '/valid')

    max_steps = 50001
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(max_steps):

            offset = (step * batch_size) % (len(train_labels) - batch_size)
            batch_data = train_set[offset:(offset + batch_size)]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # train step
            feed_dict = {'tf_input_data:0': batch_data, 'tf_labels:0': batch_labels, 'keep_prob:0': .9375}
            _, l, summary = session.run(['optimizer/train_step:0', 'loss_function/loss:0',
                                         'Merge/MergeSummary:0'], feed_dict=feed_dict)

            # get train prediction
            feed_dict = {'tf_input_data:0': batch_data, 'tf_labels:0': batch_labels, 'keep_prob:0': 1.0}
            accuracy = session.run('calculate_accuracy/accuracy:0', feed_dict=feed_dict)

            train_writer.add_summary(summary)
            if step % 500 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy)

                feed_dict = {'tf_input_data:0': valid_set, 'tf_labels:0': valid_labels, 'keep_prob:0': 1.0}
                valid_accuracy, summary = session.run(['calculate_accuracy/accuracy:0', 'Merge/MergeSummary:0'],
                                                      feed_dict=feed_dict)
                valid_writer.add_summary(summary)
                print('Validation accuracy: %.1f%%' % valid_accuracy)
                if valid_accuracy < prev_valid_accuracy:
                    break
                prev_valid_accuracy = valid_accuracy

        train_writer.flush()
        valid_writer.flush()

        # test_prediction = session.graph.get_tensor_by_name('test_prediction:0').eval()
        # print('Test accuracy: %.1f%%' % accuracy(test_prediction, test_labels))
        # visualize_some_examples(test_set, np.argmax(test_prediction, 1))
