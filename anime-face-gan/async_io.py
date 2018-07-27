import threading
import numpy as np
import os
import math


class AsyncIO:
    def __init__(self, training_set_path, batch_size):
        self._training_set_path = training_set_path
        self.batch_size = batch_size
        # async i/o thread and its variables
        self._io_thread = threading.Thread(target=self._io_thread_callback)
        self._io_finish_event = threading.Event()
        self._io_lock = threading.RLock()
        self._io_list = list()
        self._io_thread.setDaemon(True)
        self._io_max_cache_count = 5
        self._io_list_empty = threading.Event()
        self._io_list_empty.set()
        self._batch_offset = 0
        self._training_set_samples = os.listdir(self._training_set_path)
        self._training_set_samples = [os.path.join(self._training_set_path, x)
                                      for x in self._training_set_samples if x != 'count.npy']
        self._sample_count = np.load(os.path.join(training_set_path, 'count.npy'))[0]
        self._io_thread.start()

    def get_sample_count(self):
        return self._sample_count

    def _io_thread_callback(self):
        while True:
            for train_x in self._yield_training_set():
                # fetch the batch training data from disk, append to memory list
                self._io_lock.acquire()
                self._io_list.append(train_x)
                # set the empty flag to False if io list reached max_cache_count
                if len(self._io_list) >= self._io_max_cache_count:
                    self._io_list_empty.clear()
                self._io_lock.release()
                self._io_finish_event.set()

                # wait for training data being used
                self._io_list_empty.wait()

    def wait_io(self):
        self._io_finish_event.wait()
        self._io_finish_event.clear()
        self._io_lock.acquire()
        train_x = self._io_list[0][self._batch_offset: self._batch_offset + self.batch_size]
        if train_x.shape[0] < self.batch_size:
            self._io_list.pop(0)
            self._batch_offset = self.batch_size - train_x.shape[0]
            train_x2 = self._io_list[0][0: self._batch_offset]
            train_x = np.concatenate([train_x, train_x2], axis=0)
        else:
            self._batch_offset += self.batch_size
        if len(self._io_list) < self._io_max_cache_count:
            self._io_list_empty.set()
        if len(self._io_list) > 1:
            self._io_finish_event.set()
        self._io_lock.release()
        return train_x

    def _yield_training_set(self):
        random_idx = np.arange(0, len(self._training_set_samples))
        np.random.shuffle(random_idx)
        self._training_set_samples = [self._training_set_samples[i] for i in random_idx]
        for file in self._training_set_samples:
            batch_x = np.load(file)
            random_idx = np.arange(0, batch_x.shape[0])
            np.random.shuffle(random_idx)
            batch_x = [batch_x[i, :, :, :] for i in random_idx]
            batch_x = np.array(batch_x, dtype=np.uint8)
            yield np.array((batch_x - 127.5) / 127.5, dtype=np.float32)


if __name__ == '__main__':
    test = AsyncIO('animeface-np', 2)
    for x in range(int(math.ceil(test.get_sample_count() / 2))):
        result = test.wait_io()
        print(x, result[0, 0, 0, 0])
