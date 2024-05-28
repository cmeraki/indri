import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class NumpyMmapReader:
    def __init__(self, base_filename, dtype):
        self.base_filename = base_filename
        self.dtype = dtype
        self.file_index = 0
        self.element_index = 0
        self._load_file()

    def _load_file(self):
        self.current_filename = f"{self.base_filename}_{self.file_index}.npy"
        if os.path.exists(self.current_filename):

            self.mmap_file = np.memmap(self.current_filename,
                                       dtype=self.dtype,
                                       mode='r',
                                       shape=(-1,))

            self.file_size = self.mmap_file.shape[0]
        else:
            self.mmap_file = None
            self.file_size = 0

    def __iter__(self):
        self.file_index = 0
        self.element_index = 0
        self._load_file()
        return self

    def __next__(self):
        if self.mmap_file is None:
            raise StopIteration

        if self.element_index >= self.file_size:
            self.file_index += 1
            self.element_index = 0
            self._load_file()
            if self.mmap_file is None:
                raise StopIteration

        array = self.mmap_file[self.element_index]
        self.element_index += 1
        return array

    def read(self):
        return self.__next__()

    def reset(self):
        self.__iter__()


class NumpyMmapDataset(Dataset):
    def __init__(self, base_filename, dtype):
        self.reader = NumpyMmapReader(base_filename, dtype)
        self._initialize_length()

    def _initialize_length(self):
        self.length = 0
        for _ in self.reader:
            self.length += 1
        self.reader.reset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("Index out of range")

        # Reset and iterate to the desired index
        self.reader.reset()
        for i, array in enumerate(self.reader):
            if i == index:
                return array


class NumpyMmapWriter:
    def __init__(self, base_filename, dtype, elements_per_file):
        self.base_filename = base_filename
        self.dtype = dtype
        self.elements_per_file = elements_per_file
        self.current_file_index = 0
        self.current_element_index = 0
        self.mmap_file = None
        self._create_base_dir()
        self._create_new_file()

    def _create_base_dir(self):
        dir = Path(self.base_filename)
        dir.mkdir(parents=True, exist_ok=True)

    def _create_new_file(self):
        if self.mmap_file:
            self.mmap_file.flush()

        self.current_filename = f"{self.base_filename}_{self.current_file_index}.npy"
        self.current_file_index += 1
        self.mmap_file = np.memmap(self.current_filename, dtype=self.dtype, mode='w+',
                                   shape=(self.elements_per_file,))
        self.current_element_index = 0

    def write(self, array):
        if self.current_element_index >= self.elements_per_file:
            self._create_new_file()

        self.mmap_file[self.current_element_index] = array
        self.current_element_index += 1

    def close(self):
        del self.mmap_file


if __name__ == '__main__':
    # ds = NumpyMmapWriter(base_filename='test_mmap_ds/', dtype=np.int16, elements_per_file=1000)
    # for i in range(10000):
    #     x = np.array(i)
    #     ds.write(x)

    ds = NumpyMmapReader(base_filename='test_mmap_ds/', dtype=np.int16)
    for elem in ds:
        print(elem)
