from abc import abstractmethod

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


    # # Connecting layer
    # input_size = image_size
    # for layer in image_layers:
    #     input_size = layer.calculate_output_dimension(input_size)
    # connecting_layer_size = np.prod(input_size) * depths[-1]

    # fully connected layers
    # fully_connected_layers = []
    # n_nodes = [connecting_layer_size] + num_hidden
    # for i in range(len(n_nodes) - 1):
    #     fully_connected_layers.append(FullyConnectedLayer(str(layer_num), n_nodes[i], n_nodes[i + 1]))
    #     layer_num += 1
    # # classification layer
    # fully_connected_layers.append(FullyConnectedLayer(str(layer_num), n_nodes[-1], num_labels, relu_output=False))



    # # Model
    # def model(layer_input):
    #     for layer in image_layers:
    #         layer_input = layer.operate(layer_input)
    #
    #     layer_input = tf.nn.dropout(layer_input, keep_prob)
    #     shape = layer_input.get_shape().as_list()
    #     layer_input = tf.reshape(layer_input, [-1, shape[1] * shape[2] * shape[3]])
    #
    #     for layer in fully_connected_layers:
    #         layer_input = layer.operate(layer_input)
    #
    #     return layer_input
    #
    # # Training computation.
    # logits = model(tf_input)


        #######
        class EvaluationRunHook(tf.train.SessionRunHook):
            """EvaluationRunHook performs continuous evaluation of the model.
          Args:
            checkpoint_dir (string): Dir to store model checkpoints
            metric_dir (string): Dir to store metrics like accuracy and auroc
            graph (tf.Graph): Evaluation graph
            eval_frequency (int): Frequency of evaluation every n train steps
            eval_steps (int): Evaluation steps to be performed
          """

            def __init__(self, checkpoint_dir, metric_dict, graph, eval_frequency, file_path, **kwargs):

                tf.logging.set_verbosity(eval_frequency)
                self._checkpoint_dir = checkpoint_dir
                self._kwargs = kwargs
                self._eval_every = eval_frequency
                self._latest_checkpoint = None
                self._checkpoints_since_eval = 0
                self._graph = graph
                self.file_path = file_path
                # With the graph object as default graph
                # See https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
                # Adds ops to the graph object
                with graph.as_default():
                    value_dict, update_dict = tf.contrib.metrics.aggregate_metric_map(metric_dict)

                    # Op that creates a Summary protocol buffer by merging summaries
                    self._summary_op = tf.summary.merge([
                        tf.summary.scalar(name, value_op)
                        for name, value_op in value_dict.iteritems()
                    ])

                    # Saver class add ops to save and restore
                    # variables to and from checkpoint
                    self._saver = tf.train.Saver()

                    # Creates a global step to contain a counter for
                    # the global training step
                    self._gs = tf.contrib.framework.get_or_create_global_step()

                    self._final_ops_dict = value_dict
                    self._eval_ops = update_dict.values()

                # MonitoredTrainingSession runs hooks in background threads
                # and it doesn't wait for the thread from the last session.run()
                # call to terminate to invoke the next hook, hence locks.
                self._eval_lock = threading.Lock()
                self._checkpoint_lock = threading.Lock()
                # create two file writers on for training and one for validation
                self._file_writer = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'valid'))

            def after_run(self, run_context, run_values):
                # Always check for new checkpoints in case a single evaluation
                # takes longer than checkpoint frequency and _eval_every is >1
                self._update_latest_checkpoint()

                if self._eval_lock.acquire(False):
                    try:
                        if self._checkpoints_since_eval > self._eval_every:
                            self._checkpoints_since_eval = 0
                            self._run_eval()
                    finally:
                        self._eval_lock.release()

            def _update_latest_checkpoint(self):
                """Update the latest checkpoint file created in the output dir."""
                if self._checkpoint_lock.acquire(False):
                    try:
                        latest = tf.train.latest_checkpoint(self._checkpoint_dir)
                        if not latest == self._latest_checkpoint:
                            self._checkpoints_since_eval += 1
                            self._latest_checkpoint = latest
                    finally:
                        self._checkpoint_lock.release()

            def end(self, session):
                """Called at then end of session to make sure we always evaluate."""
                self._update_latest_checkpoint()

                with self._eval_lock:
                    self._run_eval()

            def _run_eval(self):
                """Run model evaluation and generate summaries."""
                coord = tf.train.Coordinator(clean_stop_exception_types=(
                    tf.errors.CancelledError, tf.errors.OutOfRangeError))

                with tf.Session(graph=self._graph) as session:

                    # Restores previously saved variables from latest checkpoint
                    self._saver.restore(session, self._latest_checkpoint)

                    session.run([
                        tf.tables_initializer(),
                        tf.local_variables_initializer()
                    ])
                    tf.train.start_queue_runners(coord=coord, sess=session)
                    train_step = session.run(self._gs)

                    tf.logging.info('Starting Evaluation For Step: {}'.format(train_step))

                    with coord.stop_on_exception():
                        eval_step = 0
                        train_set, train_labels, valid_set, valid_labels, test_set, test_labels = load_data(
                            self.file_path)
                        valid_batch_generator = BatchGenerator(valid_set, valid_labels, 500)
                        for batch_data, batch_labels in valid_batch_generator:
                            if coord.should_stop():
                                break
                            feed_dict = {'tf_input_data:0': batch_data, 'tf_labels:0': batch_labels, 'keep_prob:0': 1.0}
                            summaries, final_values, _ = session.run(
                                [self._summary_op, self._final_ops_dict, self._eval_ops], feed_dict=feed_dict)

                    # Write the summaries
                    self._file_writer.add_summary(summaries, global_step=train_step)
                    self._file_writer.flush()
                    tf.logging.info(final_values)