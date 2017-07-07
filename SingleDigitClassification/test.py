import tensorflow as tf
import numpy as np

data = np.random.rand(2, 3, 3, 3)
labels = np.zeros([2])

num_epochs = 4

queue1_input = tf.constant(data)
queue2_input = tf.constant(labels)

queue1 = tf.FIFOQueue(capacity=1, dtypes=[tf.float64, tf.float64], shapes=[(3, 3, 3), ()])
enqueue_op = queue1.enqueue_many((queue1_input, queue2_input))

close_op = queue1.close()
dequeue_op = queue1.dequeue()


queue2 = tf.FIFOQueue(capacity=200, dtypes=[tf.int32], shapes=[()])
enqueue_op2 = queue1.enqueue_many(queue1_input)
close_op2 = queue2.close()
dequeue_op2 = queue2.dequeue()


def create_session():
    config = tf.ConfigProto()
    config.operation_timeout_in_ms=2000
    return tf.InteractiveSession(config=config)


batch = tf.train.shuffle_batch(dequeue_op, batch_size=4, capacity=5, min_after_dequeue=4)

sess = create_session()
for epoch in range(num_epochs):
    sess.run(enqueue_op)
    sess.run(enqueue_op2)
sess.run(close_op)
sess.run(close_op2)


print(sess.run(dequeue_op))

####

data = [np.random.rand(3, 3, 3), np.random.rand(6, 5, 4)]
labels = np.zeros([2])

num_epochs = 4

queue1_input = [tf.constant(d) for d in data]
queue2_input = tf.constant(labels)

queue1 = tf.FIFOQueue(capacity=200, dtypes=[tf.float64, tf.float64])
enqueue_op = queue1.enqueue_many((queue1_input, queue2_input))

close_op = queue1.close()
dequeue_op = queue1.dequeue()


def create_session():
    config = tf.ConfigProto()
    config.operation_timeout_in_ms=2000
    return tf.InteractiveSession(config=config)


batch = tf.train.shuffle_batch(dequeue_op, batch_size=4, capacity=5, min_after_dequeue=4)

sess = create_session()
for epoch in range(num_epochs):
    sess.run(enqueue_op)
sess.run(close_op)


print(sess.run(dequeue_op))

