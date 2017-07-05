import numpy as np
import tensorflow as tf
from abc import abstractmethod
import multiprocessing


# def variable_summaries(var, var_name):
#     """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#     with tf.name_scope(var_name + '_summary'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))
#         tf.summary.histogram('histogram', var)


def calculate_output_dimension(input_size, patch_size, strides, padding_type):
    assert np.shape(input_size) == (2,)
    padding = np.array([0, 0])
    if padding_type == 'SAME':
        padding = (strides - 1) / 2
    output_size = (input_size - patch_size + 2 * padding) / strides + 1
    return output_size


class Layer(object):
    @abstractmethod
    def operate(self, layer_input):
        """Performs the operation defined by the layer over the layer_input variable"""


class ConvolutionLayer(Layer):
    def __init__(self, layer_num, patch_size, in_depth, out_depth, strides, padding='VALID'):
        self.layer_name = 'convolution_layer_' + layer_num
        self.patch_size = patch_size
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.strides = strides
        self.padding = padding

        self.w_shape = [patch_size[0], patch_size[1], in_depth, out_depth]
        self.b_shape = [out_depth]
        self.strides_shape = [1, strides[0], strides[1], 1]
        with tf.variable_scope(self.layer_name):
            self.w = tf.get_variable(name='weights', shape=self.w_shape,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.b = tf.Variable(tf.constant(1.0, shape=self.b_shape), name='biases')

    def operate(self, layer_input):
        with tf.variable_scope(self.layer_name + '/'):
            convolution = tf.nn.conv2d(layer_input, self.w, self.strides_shape, self.padding, name='convolution')
            return tf.nn.relu(convolution + self.b)

    def calculate_output_dimension(self, input_size):
        return calculate_output_dimension(input_size, self.patch_size, self.strides, self.padding)


# Max pooling operation with local response normalization before pooling
class MaxPoolingLayer(Layer):
    def __init__(self, layer_num, filter_size, strides, padding='SAME'):
        self.layer_name = 'max_pooling_layer_' + layer_num
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding

        self.filter_shape = [1, filter_size[0], filter_size[1], 1]
        self.strides_shape = [1, strides[0], strides[1], 1]

    def operate(self, layer_input):
        with tf.variable_scope(self.layer_name):
            lrn = tf.nn.local_response_normalization(layer_input)
            return tf.nn.max_pool(lrn, self.filter_shape, self.strides_shape, self.padding, name='P' + self.layer_name)

    def calculate_output_dimension(self, input_size):
        return calculate_output_dimension(input_size, self.filter_size, self.strides, self.padding)


# Fully connected layer = linear units plus relu after if relu_output==True
class FullyConnectedLayer(Layer):
    def __init__(self, layer_num, input_size, output_size, relu_output=True):
        self.layer_name = 'fully_connected_' + layer_num
        self.input_size = input_size
        self.output_size = output_size
        self.relu_output = relu_output

        self.w_shape = [input_size, output_size]
        self.b_shape = [output_size]
        with tf.variable_scope(self.layer_name):
            self.w = tf.get_variable(name='weights', shape=self.w_shape,
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(1.0, shape=self.b_shape), name='biases')

    def operate(self, layer_input):
        with tf.variable_scope(self.layer_name + '/'):
            output = tf.matmul(layer_input, self.w) + self.b
            if self.relu_output:
                return tf.nn.relu(output)
        return output


TRAIN, EVAL, PREDICT = 'TRAIN', 'EVAL', 'PREDICT'
CSV, EXAMPLE, JSON = 'CSV', 'EXAMPLE', 'JSON'
PREDICTION_MODES = [CSV, EXAMPLE, JSON]


def model_fn(mode):
    # this is the Udacity's course architecture
    batch_size = 64
    image_size = np.array([32, 32])
    num_channels = 1
    num_labels = 10

    # 3 convolutional layers with the same patch size and stride
    depths = [16, 32, 64]
    n = len(depths)
    patch_sizes = [np.array([5, 5])] * n
    conv_strides = [np.array([1, 1])] * n
    padding = ['VALID'] * n
    # 2 max pooling layers between the convolutional layers
    pooling_strides = [np.array([2, 2])] * (n - 1)
    pooling_filter_sizes = [np.array([2, 2])] * (n - 1)
    # one hidden layer (not including the connecting hidden layer whose size is defined by the convolutional layers)
    num_hidden = [32]

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

    # input tensors
    tf_input_data = tf.placeholder(tf.float32, shape=(None, image_size[0], image_size[1], num_channels),
                                   name='tf_input_data')
    tf_labels = tf.placeholder(tf.int64, shape=None, name='tf_labels')
    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    # Model
    def model(layer_input):
        for layer in image_layers:
            layer_input = layer.operate(layer_input)

        layer_input = tf.nn.dropout(layer_input, keep_prob)
        shape = layer_input.get_shape().as_list()
        layer_input = tf.reshape(layer_input, [-1, shape[1] * shape[2] * shape[3]])

        for layer in fully_connected_layers:
            layer_input = layer.operate(layer_input)

        return layer_input

    # Training computation.
    logits = model(tf_input_data)

    if mode in (PREDICT, EVAL):
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

    if mode in (TRAIN, EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()

    # if mode == PREDICT:
    #     # Convert predicted_indices back into strings
    #     return {
    #         'predictions': tf.gather(label_values, predicted_indices),
    #         'confidence': tf.reduce_max(probabilities, axis=1)
    #     }

    if mode == TRAIN:
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
        accuracy = tf.metrics.accuracy(tf_labels, predicted_indices)
        auc = tf.metrics.auc(labels_one_hot, probabilities)
        return {'accuracy': accuracy, 'AUC': auc}
