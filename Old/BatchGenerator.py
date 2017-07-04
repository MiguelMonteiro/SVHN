class Batch:

    def __init__(self, data, labels, batch_size):
        self.index = 0
        self.data = data
        self.labels = labels
        self.max_size = len(labels)
        self.batch_size = batch_size


    def get_next_batch(self):
        range = range(self.index,min(self.index+self.batch_size, self.max_size))
        self.index = self.index + self.batch_size
        if self.index > self.max_size:
            return [],[]
        return [self.data[i] for i in range], [self.labels[i] for i in range]

