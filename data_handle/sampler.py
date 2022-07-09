import random

from torch.utils.data import Sampler


class SequentialSampler(Sampler):
    def __init__(self, batch_size, batch_num, data_len, shuffle=False):
        super().__init__(None)
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.data_len = data_len
        self.shuffle = shuffle


    def __iter__(self):
        start_index = random.randint(0, self.data_len-1) if self.shuffle else 0
        for i in range(self.batch_num):
            end_index = start_index + self.batch_size
            if end_index > self.data_len:
                end_index = self.batch_size - (self.data_len - start_index)
                samples = list(range(start_index, self.data_len)) + list(range(end_index))
            else:
                samples = list(range(start_index, end_index))

            start_index = end_index
            if start_index >= self.data_len:
                start_index = 0

            yield samples

    def __len__(self) -> int:
        return self.batch_num

