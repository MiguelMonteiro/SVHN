import numpy as np
import tensorflow as tf
import multiprocessing

TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


# these two are only for fully connected
def xavier_normal_dist(shape):
    return tf.truncated_normal(shape, mean=0, stddev=tf.sqrt(3. / (shape[0] + shape[1])))


def xavier_uniform_dist(shape):
    lim = tf.sqrt(6. / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-lim, maxval=lim)


def convolution_layer(tf_input, patch_size, in_depth, out_depth, strides, padding='VALID'):
    w = tf.get_variable(name='weights', shape=[patch_size[0], patch_size[1], in_depth, out_depth],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.Variable(tf.constant(1.0, shape=[out_depth]), name='biases')
    convolution = tf.nn.conv2d(tf_input, w, strides, padding)
    return convolution + b


def max_pooling_layer(tf_input, filter_size, strides, padding='SAME'):
    lrn = tf.nn.local_response_normalization(tf_input)
    return tf.nn.max_pool(lrn, [1, filter_size[0], filter_size[1], 1], strides, padding)


def fully_connected_layer(tf_input, input_size, output_size):
    w = tf.Variable(xavier_uniform_dist([input_size, output_size]), name='weights')
    b = tf.Variable(tf.constant(1.0, shape=[output_size]), name='biases')
    return tf.matmul(tf_input, w) + b


def model_fn(mode, tf_input, tf_labels):
    image_size = [32, 32]
    num_channels = 1
    # input tensors
    #tf_input = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], num_channels), name='tf_input_data')
    #tf_labels = tf.placeholder(tf.int64, shape=None, name='tf_labels')

    if mode is TRAIN:
        keep_prob = .9375
    else:
        keep_prob = 1.

    # this is the Udacity's course architecture
    num_channels = 1
    num_labels = 10

    # 3 convolutional layers with the same patch size and stride
    depths = [16, 32, 64]
    n = len(depths)
    num_hidden = [32]

    tensor = tf_input
    depths = [num_channels] + depths
    for i in range(len(depths) - 1):
        with tf.variable_scope('layer_{0}'.format(i)):
            if i > 0:
                tensor = max_pooling_layer(tensor, [2, 2], [1, 2, 2, 1])
            tensor = tf.nn.relu(convolution_layer(tensor, [5, 5], depths[i], depths[i + 1], [1, 1, 1, 1]))

    tensor = tf.nn.dropout(tensor, keep_prob)
    shape = tensor.get_shape().as_list()
    tensor = tf.reshape(tensor, [-1, shape[1] * shape[2] * shape[3]])

    tensor = tf.nn.relu(fully_connected_layer(tensor, shape[-1], num_hidden[0]))

    logits = fully_connected_layer(tensor, num_hidden[-1], num_labels)

    # compute probabilities regardless of mode
    probabilities = tf.nn.softmax(logits)
    predicted_indices = tf.argmax(probabilities, 1)
    # this maintains a running accuracy (acc, acc_op)
    accuracy = tf.metrics.accuracy(tf_labels, predicted_indices)

    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

    # if mode == PREDICT:
    #     # Convert predicted_indices back into strings
    #     return {
    #         'predictions': tf.gather(label_values, predicted_indices),
    #         'confidence': tf.reduce_max(probabilities, axis=1)
    #     }

    if mode == TRAIN:
        # save running accuracy even in training
        tf.summary.scalar('train_accuracy', accuracy[0])

        # calculate cross entropy loss
        with tf.variable_scope('loss_function'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels))
            tf.summary.scalar('loss', loss)
        # Optimizer.
        with tf.variable_scope('optimizer'):
            learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
            train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)
        return train_op, global_step

    if mode == EVAL:
        labels_one_hot = tf.one_hot(tf_labels, depth=num_labels, on_value=True, off_value=False, dtype=tf.bool)
        auc = tf.metrics.auc(labels_one_hot, probabilities)
        return {'accuracy': accuracy, 'AUC': auc}


def parse_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        })

    with tf.variable_scope('decoder'):
        image = tf.decode_raw(features['img_raw'], tf.float32)
        label = tf.cast(features['label'], tf.int32)

    with tf.variable_scope('image'):
        # reshape and add the channel dimension
        image = tf.expand_dims(tf.reshape(image, [32, 32]), -1)

    return image, label


def input_fn(filenames, num_epochs=None, shuffle=True, batch_size=64):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    image, label = parse_example(example)

    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size,
            min_after_dequeue=2 * batch_size + 1,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=False,
            allow_smaller_final_batch=True
        )
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size,
            capacity=batch_size * 10,
            num_threads=multiprocessing.cpu_count(),
            enqueue_many=False,
            allow_smaller_final_batch=True
        )

    return images, labels
