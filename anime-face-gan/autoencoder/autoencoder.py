import os
import keras.backend as k
from keras.layers import Conv2D, Input, Dense, Reshape, LeakyReLU, Flatten, Cropping2D
from keras.optimizers import Adam
from keras.models import Model
from train_model_cli_dcgan_v2 import set_vram_growth
from pixel_shuffler import PixelShuffler
from cv2 import imdecode, imread, imwrite, IMREAD_COLOR
from math import sqrt, ceil
from random import randint
import threading
from tqdm import tqdm
import numpy as np


def keras_set_vram_growth():
    sess = set_vram_growth()
    k.set_session(sess)


class AutoEncoder:
    def __init__(self, input_shape, training_set_path, target_dims=1024, batch_size=32):
        self._input_shape = input_shape
        self._target_dims = target_dims
        self._encoder = None
        self._decoder = None
        self._encoder = self._get_encoder()
        self._decoder = self._get_decoder()
        self._autoencoder = self._get_autoencoder()
        self._opt = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        self._autoencoder.compile(self._opt, loss='mean_absolute_error')
        self._encoder.summary()
        self._decoder.summary()
        self._batch_size = batch_size
        self._training_set_path = training_set_path
        # saving avg and var for training data
        self._training_set_dist = None
        # async i/o thread and its variables
        self._training_set_samples = os.listdir(self._training_set_path)
        self._training_set_samples = [os.path.join(self._training_set_path, x) for x in self._training_set_samples]
        print('Loading training set')
        self._training_set_samples = [self._load_data(x) for x in tqdm(self._training_set_samples, ascii=True)]
        self._io_thread = threading.Thread(target=self._io_thread_callback)
        self._io_finish_event = threading.Event()
        self._io_lock = threading.RLock()
        self._io_list = list()
        self._io_thread.setDaemon(True)
        self._io_max_cache_count = 30
        self._io_list_empty = threading.Event()
        self._io_list_empty.set()
        self._io_thread.start()

    @staticmethod
    def _load_data(path):
        with open(path, 'rb') as f:
            length = f.seek(0, 2)
            f.seek(0)
            raw_data = f.read(length)
            return raw_data

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        input_shape = self._input_shape
        target_dims = self._target_dims
        input_ = Input(input_shape)
        model = self._conv(64)(input_)
        model = self._conv(128)(model)
        model = self._conv(256)(model)
        model = self._conv(512)(model)

        model = Flatten()(model)
        model = Dense(target_dims, activation='tanh')(model)

        model = Model(input_, model)
        if os.path.exists('encoder.h5'):
            model.load_weights('encoder.h5')
        return model

    def _get_decoder(self):
        if self._decoder is not None:
            return self._decoder
        input_shape = (self._target_dims,)
        output_dims = self._input_shape
        input_ = Input(input_shape)
        h, w, c = output_dims
        h2, w2 = int(ceil(h / 2)), int(ceil(w / 2))
        h4, w4 = int(ceil(h2 / 2)), int(ceil(w2 / 2))
        h8, w8 = int(ceil(h4 / 2)), int(ceil(w4 / 2))
        h16, w16 = int(ceil(h8 / 2)), int(ceil(w8 / 2))
        model = Dense(h16 * w16 * 512)(input_)
        model = Reshape((h16, w16, 512))(model)
        model = self._upscale((h8, w8, 256))(model)
        model = self._upscale((h4, w4, 128))(model)
        model = self._upscale((h2, w2, 64))(model)
        model = self._upscale((h, w, 32))(model)
        # the following layer just mapped to [-1, 1] range, possible occurs gradient loss
        model = Conv2D(c, 5, padding='same', activation='tanh')(model)

        model = Model(input_, model)
        if os.path.exists('decoder.h5'):
            model.load_weights('decoder.h5')
        return model

    def _get_autoencoder(self):
        input_ = Input(self._input_shape)
        encoder = self._encoder(input_)
        decoder = self._decoder(encoder)
        return Model(input_, decoder)

    @staticmethod
    def _conv(filters):
        def block(x):
            x = Conv2D(filters, 5, strides=2, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            return x

        return block

    @staticmethod
    def _upscale(out_shape):
        def block(x):
            x = Conv2D(out_shape[2] * 4, 3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            shape = x.get_shape().as_list()[1:]
            cx = shape[0] - out_shape[0]
            cy = shape[1] - out_shape[1]
            if cx != 0 or cy != 0:
                x = Cropping2D(((0, cx), (0, cy)))(x)
            return x

        return block

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

    def _wait_io(self):
        while True:
            self._io_finish_event.wait()
            self._io_finish_event.clear()
            self._io_lock.acquire()
            train_x = self._io_list.pop()
            if len(self._io_list) < self._io_max_cache_count:
                self._io_list_empty.set()
            self._io_lock.release()
            yield train_x, train_x

    def _yield_training_set(self):
        _cache_files = list()
        random_idx = np.arange(0, len(self._training_set_samples))
        np.random.shuffle(random_idx)
        self._training_set_samples = [self._training_set_samples[i] for i in random_idx]
        for file in self._training_set_samples:
            # image = imread(file)
            image = imdecode(np.asarray(bytearray(file), dtype=np.uint8), IMREAD_COLOR)
            _cache_files.append(image)

            if len(_cache_files) == self._batch_size:
                batch_x = np.array(_cache_files, dtype=np.float32)
                _cache_files = list()
                yield (batch_x - 127.5) / 127.5
        if len(_cache_files) > 0:
            batch_x = np.array(_cache_files, dtype=np.float32)
            yield (batch_x - 127.5) / 127.5

    def train(self, epochs=100):
        # self._autoencoder.fit(train_data, train_data, batch_size=self._batch_size, epochs=epochs, shuffle=True)
        self._autoencoder.fit_generator(self._wait_io(), int(ceil(len(self._training_set_samples) / self._batch_size)),
                                        epochs=epochs)
        self._encoder.save_weights('encoder.h5')
        self._decoder.save_weights('decoder.h5')

    def test_gen(self, output_file, sample_count=100, batch_size=None):
        wc = int(ceil(sqrt(sample_count)))
        hc = int(ceil(sample_count / wc))
        if self._training_set_dist is None:
            _noise = np.random.normal(0, 0.2, [sample_count, self._target_dims])
        else:
            _noise = np.zeros([sample_count, self._target_dims])
            for i in range(self._target_dims):
                avg = self._training_set_dist['avg'][i]
                var = self._training_set_dist['var'][i]
                _noise[:, i] = np.random.normal(avg, np.sqrt(var), (sample_count,))
        _noise = np.clip(_noise, -1, 1)

        results = self._decoder.predict(_noise, batch_size=batch_size)
        ret = np.zeros((wc * self._input_shape[0], hc * self._input_shape[1], self._input_shape[2]))
        for i, result in zip(range(sample_count), results):
            h = i // wc
            w = i % wc
            ret[h * self._input_shape[0]:(h + 1) * self._input_shape[0],
                w * self._input_shape[1]:(w + 1) * self._input_shape[1], :] = result
        imwrite(output_file, (ret * 127.5 + 127.5).astype(np.uint8))

    def test_train(self, train_path, output_file, sample_count=100, batch_size=None):
        wc = int(ceil(sqrt(sample_count)))
        hc = int(ceil(sample_count / wc))
        files = os.listdir(train_path)
        idx = np.arange(len(files))
        np.random.shuffle(idx)
        files = [files[i] for i in idx]
        file_in = []
        for file in files:
            img = imread(os.path.join(train_path, file))
            file_in.append(img)
            if len(file_in) == sample_count:
                break
        file_in = np.array(file_in, dtype=np.float32)
        file_in = (file_in - 127.5) / 127.5
        noise = self._encoder.predict(file_in, batch_size=batch_size)
        results = self._decoder.predict(noise, batch_size=batch_size)
        ret = np.zeros((wc * self._input_shape[0], hc * self._input_shape[1], self._input_shape[2]))
        for i, result in zip(range(sample_count), results):
            h = i // wc
            w = i % wc
            ret[h * self._input_shape[0]:(h + 1) * self._input_shape[0],
                w * self._input_shape[1]:(w + 1) * self._input_shape[1], :] = result
        imwrite(output_file, (ret * 127.5 + 127.5).astype(np.uint8))

    def test_scale(self, output_file, first_dim=None, second_dim=None, sample_count=100, batch_size=None):
        wc = int(ceil(sqrt(sample_count)))
        hc = int(ceil(sample_count / wc))

        _noise = np.zeros([wc, hc, self._target_dims])
        if first_dim is None:
            first_dim = randint(0, self._target_dims - 1)
        if second_dim is None:
            second_dim = randint(0, self._target_dims - 2)
            if second_dim >= first_dim:
                second_dim += 1
        _noise[:, :, first_dim] = np.linspace(-1, 1, wc)
        _noise[:, :, second_dim] = np.linspace(-1, 1, hc)
        _noise[:, :, second_dim] = _noise[:, :, second_dim].T
        _noise = _noise.reshape((-1, self._target_dims))
        results = self._decoder.predict(_noise, batch_size=batch_size)
        ret = np.zeros((wc * self._input_shape[0], hc * self._input_shape[1], self._input_shape[2]))
        for i, result in zip(range(sample_count), results):
            h = i // wc
            w = i % wc
            ret[h * self._input_shape[0]:(h + 1) * self._input_shape[0],
                w * self._input_shape[1]:(w + 1) * self._input_shape[1], :] = result
        imwrite(output_file, (ret * 127.5 + 127.5).astype(np.uint8))

    def calc_training_set_vector(self):
        vectors = np.zeros((len(self._training_set_samples), self._target_dims), dtype=np.float32)
        gen = self._wait_io()
        _sum = 0
        for _ in tqdm(range(int(ceil(len(self._training_set_samples) / self._batch_size))), ascii=True):
            batch, _ = next(gen)
            next_sum = _sum + batch.shape[0]
            vectors[_sum:next_sum, :] = self._encoder.predict(batch, batch_size=self._batch_size)
            _sum = next_sum
        avg = np.average(vectors, 0)
        var = np.var(vectors, 0)
        self._training_set_dist = {'avg': avg, 'var': var}


class AsyncInput:
    def __init__(self):
        self._is_stop = False
        self._thd = threading.Thread(target=self._cb)
        self._thd.setDaemon(True)
        self._thd.start()

    def is_stop(self):
        return self._is_stop

    def _cb(self):
        _ = input('')
        self._is_stop = True


def main(eval_only=False):
    keras_set_vram_growth()
    nn = AutoEncoder((100, 100, 3), 'train', 128, 16)
    if eval_only:
        nn.calc_training_set_vector()
        evaluate(nn)
        return
    print('Press "Enter" to exit once reaching 5 epochs end')
    interrupter = AsyncInput()
    while not interrupter.is_stop():
        nn.train(5)
        evaluate(nn)


def evaluate(nn):
    nn.test_gen('test.png', batch_size=16)
    nn.test_train('train', 'test2.png', batch_size=16)
    nn.test_scale('test3.png', batch_size=16)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        main(True)
    else:
        main(False)
