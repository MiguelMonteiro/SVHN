import tensorflow as tf


def get_graph(image_size, batch_size, num_channels, num_labels, valid_set, test_set):
    patch_size = 5
    depth1 = 16
    depth2 = 32
    depth3 = 64
    num_hidden1 = 64
    num_hidden2 = 16
    shape = [batch_size, image_size, image_size, num_channels]

    # Construct a 7-layer CNN.
    # C1: convolutional layer, batch_size x 28 x 28 x 16, convolution size: 5 x 5 x 1 x 16
    # S2: sub-sampling layer, batch_size x 14 x 14 x 16
    # C3: convolutional layer, batch_size x 10 x 10 x 32, convolution size: 5 x 5 x 16 x 32
    # S4: sub-sampling layer, batch_size x 5 x 5 x 32
    # C5: convolutional layer, batch_size x 1 x 1 x 64, convolution size: 5 x 5 x 32 x 64
    # Dropout
    # F6: fully-connected layer, weight size: 64 x 16
    # Output layer, weight size: 16 x 10

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        tf_train_set = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),
                                      name='tf_train_set')
        tf_train_labels = tf.placeholder(tf.int64, shape=(batch_size), name='tf_train_labels')
        tf_valid_set = tf.constant(valid_set)
        tf_test_set = tf.constant(test_set)

        # Variables.
        l1w = tf.get_variable("W1", shape=[patch_size, patch_size, num_channels, depth1],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
        l1b = tf.Variable(tf.constant(1.0, shape=[depth1]), name='B1')

        l2w = tf.get_variable("W2", shape=[patch_size, patch_size, depth1, depth2],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
        l2b = tf.Variable(tf.constant(1.0, shape=[depth2]), name='B2')

        l3w = tf.get_variable("W3", shape=[patch_size, patch_size, depth2, num_hidden1],
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
        l3b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]), name='B3')

        l4w = tf.get_variable("W4", shape=[num_hidden1, num_hidden2],
                              initializer=tf.contrib.layers.xavier_initializer())
        l4b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]), name='B4')

        l5w = tf.get_variable("W5", shape=[num_hidden2, num_labels],
                              initializer=tf.contrib.layers.xavier_initializer())
        l5b = tf.Variable(tf.constant(1.0, shape=[num_labels]), name='B5')

        # Model.
        def model(data, keep_prob, shape):
            # LCN = LecunLCN(data, shape)
            LCN = data
            conv = tf.nn.conv2d(LCN, l1w, [1, 1, 1, 1], 'VALID', name='C1')
            hidden = tf.nn.relu(conv + l1b)
            lrn = tf.nn.local_response_normalization(hidden)
            sub = tf.nn.max_pool(lrn, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S2')

            conv = tf.nn.conv2d(sub, l2w, [1, 1, 1, 1], padding='VALID', name='C3')
            hidden = tf.nn.relu(conv + l2b)
            lrn = tf.nn.local_response_normalization(hidden)
            sub = tf.nn.max_pool(lrn, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S4')

            conv = tf.nn.conv2d(sub, l3w, [1, 1, 1, 1], padding='VALID', name='C5')
            hidden = tf.nn.relu(conv + l3b)

            hidden = tf.nn.dropout(hidden, keep_prob)
            shape = hidden.get_shape().as_list()
            reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

            hidden = tf.nn.relu(tf.matmul(reshape, l4w) + l4b)
            return tf.matmul(hidden, l5w) + l5b

        # Training computation.
        logits = model(tf_train_set, 0.9375, shape)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels),
                              name='loss')

        # Optimizer.
        # optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step,
                                                                      name='optimizer')

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(model(tf_train_set, 1.0, shape), name='train_prediction')
        valid_prediction = tf.nn.softmax(model(tf_valid_set, 1.0, shape), name='valid_prediction')
        test_prediction = tf.nn.softmax(model(tf_test_set, 1.0, shape), name='test_prediction')

        saver = tf.train.Saver()

    return graph
