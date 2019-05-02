import math
import numpy as np


class DataManager(object):
    def __init__(self, data, num_epoch, batch_size, *, shuffle=True, align=False, simple=True, infinite=False):
        self.data = data
        self.data_length = len(data)
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.cur_epoch = 1
        self.cur_batch = 1
        self.cur_pos = 0
        self.num_batch = int(math.ceil(float(self.data_length)/batch_size))
        self.simple = simple
        self.align = align
        self.infinite = infinite

        self.data_index = []
        res = batch_size - (self.data_length % batch_size)
        if shuffle:
            np.random.shuffle(self.data)

        for i in range(num_epoch):
            if shuffle:
                ids = list(np.random.permutation(self.data_length))
            else:
                ids = [d for d in range(self.data_length)]
            if align and res != 0:
                add_on = ids[:res]
                ids.extend(add_on)
            self.data_index.extend(ids)

    def get_batch(self):
        if self.infinite:
            if self.data_length < self.batch_size:
                replace = True
            else:
                replace = False
            idx = list(np.random.choice(self.data_length, self.batch_size, replace=replace))
            return [self.data[i] for i in idx]

        if self.simple:
            batch = self.data[(self.cur_batch-1)*self.batch_size: self.cur_batch*self.batch_size]
            if self.align and len(batch) < self.batch_size:
                batch = batch + batch[: self.batch_size-len(batch)]
            self.cur_pos += self.batch_size
        else:
            start = self.cur_pos
            end = self.cur_pos + self.batch_size - 1
            batch = []
            while start != end + 1 and start < len(self.data_index):
                batch.append(self.data[self.data_index[start]])
                start += 1
            self.cur_pos = end + 1

        self.cur_batch += 1
        if (self.cur_batch-1) % self.num_batch == 0:
            self.cur_epoch += 1
            self.cur_batch = 1
        return batch


if __name__ == '__main__':
    data = [i for i in range(12)]
    dm = DataManager(data, 2, 5)
    print(dm.simple)
    for i in range(dm.num_epochs):
        for j in range(dm.num_batch):
            print('epoch: %d, batch: %d'%(dm.cur_epoch, dm.cur_batch))
            batch = dm.get_batch()
            print(batch)
