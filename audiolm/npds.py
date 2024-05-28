import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import uuid


class NumpyDataset:
    def __init__(self,
                 dir,
                 samples_per_file=None,
                 dtype=None):

        self.dir = dir
        self.samples_per_file = samples_per_file
        self.dtype = dtype

        self.index = 0
        self.array = None

    def new_array(self):
        self.index = 0
        self.array = np.zeros(self.samples_per_file, dtype=self.dtype)

    def flush(self):
        if self.array is None:
            return

        self.array = self.array[:self.index]
        file_prefix = str(uuid.uuid4())
        file_path = f"{self.dir}/{file_prefix}.npy"
        np.save(file_path, self.array)

    def write(self, samples):
        if self.array is None:
            self.new_array()

        if len(samples) + self.index > self.samples_per_file:
            self.flush()
            self.new_array()

        self.array[self.index:self.index + len(samples)] = samples
        self.index += len(samples)

    def close(self):
        self.flush()

if __name__ == '__main__':

    ds = NumpyDataset(dir='test_mmap_ds/',
                      samples_per_file=1000000,
                      dtype=np.int32)

    total_samples = 0

    for i in range(100000):
        x = np.random.randn(np.random.randint(10,100))
        total_samples += len(x)
        ds.write(x)

    ds.close()
    print(total_samples)