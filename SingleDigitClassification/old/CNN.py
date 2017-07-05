import numpy as np
import tensorflow as tf

from SingleDigitClassification.old.Layers import ConvolutionLayer, MaxPoolingLayer, FullyConnectedLayer


def build_architecture(image_size, num_channels, num_labels, patch_sizes, depths, conv_strides,
                       pooling_filter_sizes, pooling_strides, num_hidden):
    layer_num = 1

    # Image Layers
    image_layers = []
    depths = [num_channels] + depths
    for i in range(len(depths) - 1):
        if i > 0:
            image_layers.append(MaxPoolingLayer(str(layer_num), pooling_filter_sizes[i - 1], pooling_strides[i - 1]))
        image_layers.append(ConvolutionLayer(str(layer_num), patch_sizes[i], depths[i], depths[i + 1], conv_strides[i]))
        layer_num += 1

    # Connecting layer
    input_size = image_size
    for layer in image_layers:
        input_size = layer.calculate_output_dimension(input_size)
    connecting_layer_size = np.prod(input_size) * depths[-1]

    # fully connected layers
    fully_connected_layers = []
    n_nodes = [connecting_layer_size] + num_hidden
    for i in range(len(n_nodes) - 1):
        fully_connected_layers.append(FullyConnectedLayer(str(layer_num), n_nodes[i], n_nodes[i + 1]))
        layer_num += 1
    # classification layer
    fully_connected_layers.append(FullyConnectedLayer(str(layer_num), n_nodes[-1], num_labels, relu_output=False))
    return image_layers, fully_connected_layers


def calc_accuracy(labels, prediction):
    with tf.variable_scope('calculate_accuracy'):
        return tf.reduce_mean(100*tf.cast(tf.equal(labels, tf.argmax(prediction, 1)), dtype=tf.float32), name='accuracy')


# Convolutional layers with max pooling in between, dropout before connecting layer, followed by fully connected
class ConventionalCNN(object):
    def __init__(self, image_size, batch_size, num_channels, num_labels, patch_sizes, depths, conv_strides,
                 pooling_filter_sizes, pooling_strides, num_hidden):
        assert len(depths) == len(conv_strides)
        assert len(depths) == len(patch_sizes)
        assert len(pooling_strides) == len(pooling_filter_sizes)
        assert len(pooling_strides) == len(depths) - 1

        self.image_layers, self.fully_connected_layers = \
            build_architecture(image_size, num_channels, num_labels, patch_sizes, depths, conv_strides,
                               pooling_filter_sizes, pooling_strides, num_hidden)

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_labels = num_labels

    def get_graph(self):
        graph = tf.Graph()

        with graph.as_default():
            # Input data
            tf_input_data = tf.placeholder(tf.float32, shape=(None, self.image_size[0], self.image_size[1],
                                                              self.num_channels), name='tf_input_data')
            tf_labels = tf.placeholder(tf.int64, shape=None, name='tf_labels')

            keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

            # Initialize variables
            for layer in self.image_layers + self.fully_connected_layers:
                layer.initialize_tensors()

            # Model
            def model(data):
                layer_input = data
                for i, layer in enumerate(self.image_layers):
                    layer_input = layer.operate(layer_input)

                layer_input = tf.nn.dropout(layer_input, keep_prob)
                shape = layer_input.get_shape().as_list()
                layer_input = tf.reshape(layer_input, [-1, shape[1] * shape[2] * shape[3]])

                for i, layer in enumerate(self.fully_connected_layers):
                    layer_input = layer.operate(layer_input)

                return layer_input

            # Training computation.
            logits = model(tf_input_data)
            with tf.variable_scope('loss_function'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels),
                                      name='loss')

            # Optimizer.
            # optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
            with tf.variable_scope('optimizer'):
                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
                train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                               name='train_step')

            # Predictions for the training, validation, and test data.
            prediction = tf.nn.softmax(logits, name='prediction')
            accuracy = calc_accuracy(tf_labels, prediction)

            tf.summary.scalar('metrics/loss', loss)
            tf.summary.scalar('metrics/global_step', global_step)
            tf.summary.scalar('metrics/learning_rate', learning_rate)
            tf.summary.scalar('metrics/accuracy', accuracy)

            summary = tf.summary.merge_all()

            saver = tf.train.Saver()

        return graph
