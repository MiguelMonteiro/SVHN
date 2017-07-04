import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

image_size = 32
num_labels = 10
num_channels = 1

filename = 'sets.pickle'


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


def load_data(filename):
    with open(filename, "rb") as f:
        train_set =  pickle.load(f)
        train_labels = pickle.load(f)
        valid_set = pickle.load(f)
        valid_labels = pickle.load(f)
        test_set = pickle.load(f)
        test_labels = pickle.load(f)
        train_set, train_labels = reformat(train_set, train_labels)
        valid_set, valid_labels = reformat(valid_set, valid_labels)
        test_set, test_labels = reformat(test_set, test_labels)
        return train_set, train_labels, valid_set, valid_labels, test_set, test_labels


train_set, train_labels, valid_set, valid_labels, test_set, test_labels = load_data(filename)


graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(1)
    tf_train_dataset = tf.constant(train_set, dtype=tf.float32)
    tf_train_labels = tf.constant(train_labels, dtype=tf.float32)
    tf_valid_dataset = tf.constant(valid_set, dtype=tf.float32)
    tf_test_dataset = tf.constant(test_set, dtype=tf.float32)


    # Variables.
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]), dtype=tf.float32)
    biases = tf.Variable(tf.zeros([num_labels]), dtype=tf.float32)

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(1).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 2001


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 50 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

    print('Test accuracy: %.3f%%' % accuracy(test_prediction.eval(), test_labels))
    # should be 0.931818181818
