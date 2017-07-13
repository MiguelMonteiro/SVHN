from task import run

run('', True, 1000, 1000, 'logs', ['data/train_data.tfrecord'], ['data/valid_data.tfrecord'], 64, 64, 0.01, 1000, 10, 'sdf')

