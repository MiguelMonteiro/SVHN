from abc import abstractmethod
import tensorflow as tf
import numpy as np


def variable_summaries(var, var_name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(var_name + '_summary'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def calculate_output_dimension(input_size, patch_size, strides, padding_type):
    assert np.shape(input_size) == (2,)
    padding = np.array([0, 0])
    if padding_type == 'SAME':
        padding = (strides - 1) / 2
    output_size = (input_size - patch_size + 2 * padding) / strides + 1
    return output_size


class Layer(object):
    @abstractmethod
    def initialize_tensors(self):
        """Initializes the tensors needed for operating the layer, must be done with graph open"""

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
        self.w = None
        self.b = None

    def initialize_tensors(self):
        with tf.variable_scope(self.layer_name):
            self.w = tf.get_variable(name='weights', shape=self.w_shape,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.b = tf.Variable(tf.constant(1.0, shape=self.b_shape), name='biases')
            variable_summaries(self.w, 'weights')
            variable_summaries(self.b, 'biases')

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

    def initialize_tensors(self):
        pass

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
        self.w = None
        self.b = None

    def initialize_tensors(self):
        with tf.variable_scope(self.layer_name):
            self.w = tf.get_variable(name='weights', shape=self.w_shape,
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(1.0, shape=self.b_shape), name='biases')
            variable_summaries(self.w, 'weights')
            variable_summaries(self.b, 'biases')

    def operate(self, layer_input):
        with tf.variable_scope(self.layer_name + '/'):
            output = tf.matmul(layer_input, self.w) + self.b
            if self.relu_output:
                return tf.nn.relu(output)
        return output
