import os
import math
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io
import threading
from async_io import AsyncIO


def set_vram_growth(as_default_sess=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if as_default_sess:
        return tf.InteractiveSession(config=config)
    else:
        return tf.Session(config=config)


def rescale_to_rgb(image):
    return (image + 1) / 2


# tensorflow util functions
def conv2d(i, output_dim, kernel_size=(5, 5), strides=(2, 2), stddev=0.02, name='conv2d'):
    (k_h, k_w), (s_h, s_w) = kernel_size, strides
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, i.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(i, w, strides=[1, s_h, s_w, 1], padding='SAME')
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
    return conv


def deconv2d(i, output_shape, kernel_size=(5, 5), strides=(2, 2), stddev=0.02, name='deconv2d', output_weights=False):
    (k_h, k_w), (s_h, s_w) = kernel_size, strides
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], i.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if output_shape[0]:
            deconv = tf.nn.conv2d_transpose(i, w, output_shape=output_shape, strides=[1, s_h, s_w, 1])
        else:
            deconv = tf.nn.conv2d_transpose(i, w, output_shape=[tf.shape(i)[0]] + output_shape[1:],
                                            strides=[1, s_h, s_w, 1])
            deconv = tf.reshape(deconv, [-1] + output_shape[1:])
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, b)
    if output_weights:
        return deconv, w, b
    else:
        return deconv


def leaky_relu(x, alpha=0.2, name='leaky_relu'):
    with tf.variable_scope(name):
        return tf.maximum(x, alpha * x)


def dense(i, output_dim, name='linear', stddev=0.02, use_bias=True):
    shape = i.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
        if use_bias:
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
            mul = tf.matmul(i, w) + b
        else:
            mul = tf.matmul(i, w)
        return mul


def batch_norm(i, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(i, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True,
                                        is_training=train, scope=name)


def calc_conv_out_shape_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def save_images(output_name, images, swap_rgb=False):
    m, h, w, c = images.shape
    rows = int(math.ceil(math.sqrt(m)))
    cols = rows
    out_image = np.zeros((rows * h, cols * w, c))
    for y in range(rows):
        for x in range(cols):
            offset = y * cols + x
            if offset >= m:
                continue
            out_image[y * h:(y + 1) * h, x * w:(x + 1) * w, :] = images[offset]
    if swap_rgb:
        # swapping RGB channel
        t = out_image.copy()[:, :, 0]
        out_image[:, :, 0] = out_image[:, :, 2]
        out_image[:, :, 2] = t
    io.imsave(output_name, out_image)


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


def set_exists(set_, item):
    if type(item) == set:
        return len(set_.intersection(item)) > 0
    return len(set_.intersection({item})) > 0


class WGAN:

    def __init__(self, training_set_path, batch_size=128, image_width=100, image_height=100, channel_count=3,
                 d_filter=64, g_filter=64, epochs=100, d_arch={'bn_first'}, g_arch={'bn_first'}, noise_dim=100,
                 d_adam_lr=0.0002, d_adam_b1=0.5, g_adam_lr=0.0002, g_adam_b1=0.5, dropout_rate=0.0,
                 test_generator_per_step=100, save_model_per_step=500, weight_dir='model', log_dir='log',
                 output_dir='output', sample_count=64, random_count=64, d_opt_runs_per_step=1, g_opt_runs_per_step=1):
        self.training_set_path = training_set_path
        self.image_width = image_width
        self.image_height = image_height
        self.channel_count = channel_count
        self.d_filter = d_filter
        self.d_arch = d_arch
        self.g_filter = g_filter
        self.g_arch = g_arch
        self._step = 0
        self._summary_dict = dict()
        self.noise_dim = noise_dim
        self.d_adam_lr = d_adam_lr
        self.d_adam_beta1 = d_adam_b1
        self.g_adam_lr = g_adam_lr
        self.g_adam_beta1 = g_adam_b1
        self.dropout_rate = dropout_rate
        self.test_generator_per_step = test_generator_per_step
        self.save_model_per_step = save_model_per_step
        self.weight_dir = weight_dir
        self.log_dir = log_dir
        self.output_dir = output_dir
        self.d_opt_runs_per_step = d_opt_runs_per_step
        self.g_opt_runs_per_step = g_opt_runs_per_step
        self.epochs = epochs
        self.sample_count = sample_count
        self.random_count = random_count
        self.batch_size = batch_size
        self._async_io = AsyncIO(training_set_path, batch_size)

        # validation test for architecture string
        if not set_exists(self.d_arch, {'selu', 'bn_first', 'bn_last', 'resnet'}):
            raise ValueError('d_arch should be one of "selu", "bn_first", "bn_last" or "resnet"')
        if not set_exists(self.g_arch, {'selu', 'bn_first', 'bn_last', 'resnet'}):
            raise ValueError('g_arch should be one of "selu", "bn_first", "bn_last" or "resnet"')

        # validation test for traning set path
        if not os.path.exists(self.training_set_path):
            raise ValueError('IO check: training set path not exists')

        self._input_tensor = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.channel_count],
                                            name='discriminator/input')
        self._noise_tensor = tf.placeholder(tf.float32, [None, self.noise_dim], name='generator/input')
        self._dropout_tensor = tf.placeholder(tf.float32, name='dropout_rate')
        self._d_logits = self._discriminator_model(self._input_tensor)
        self._g = self._generator_model()
        self._gan_logits = self._discriminator_model(self._g, reuse=True)
        self.sampler = self._generator_model(reuse=True, train=False)

        # generating the loss function
        self._d_loss, self._g_loss = self._compile_loss()

        self._t_vars = tf.trainable_variables()
        self._d_vars = [var for var in self._t_vars if 'discriminator' in var.name]
        self._g_vars = [var for var in self._t_vars if 'generator' in var.name]
        self._d_opt = tf.train.AdamOptimizer(self.d_adam_lr, beta1=self.d_adam_beta1)\
            .minimize(self._d_loss, var_list=self._d_vars, colocate_gradients_with_ops=True)
        self._g_opt = tf.train.AdamOptimizer(self.g_adam_lr, beta1=self.g_adam_beta1)\
            .minimize(self._g_loss, var_list=self._g_vars, colocate_gradients_with_ops=True)

        tf.contrib.slim.model_analyzer.analyze_vars(self._t_vars, print_info=True)

        # set the sample noise using the specified seed
        np.random.seed(0)
        self._sample_noise = np.random.uniform(-1.0, 1.0, size=[sample_count, noise_dim])
        import time
        t = int(time.time())
        np.random.seed(t)

        self._summary_dict['d_pred'] = \
            tf.summary.histogram('d_pred', tf.reshape(tf.concat([self._d_logits, self._gan_logits], axis=0), [1, -1]))
        self._summary_dict['g_pred'] = tf.summary.image('g_pred', self._g, max_outputs=random_count)
        self._summary_dict['g_trace'] = tf.summary.image('g_trace', self._g, max_outputs=sample_count)
        self._summary_dict['d_lr'] = tf.summary.scalar('d_lr', self.d_adam_lr)
        self._summary_dict['d_beta1'] = tf.summary.scalar('d_beta1', self.d_adam_beta1)
        self._summary_dict['g_lr'] = tf.summary.scalar('g_lr', self.g_adam_lr)
        self._summary_dict['g_beta1'] = tf.summary.scalar('g_beta1', self.g_adam_beta1)

        # starting the session
        self._saver = tf.train.Saver()
        self._sess = set_vram_growth()
        tf.global_variables_initializer().run(session=self._sess)

        # creating new directory
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.weight_dir):
            os.mkdir(self.weight_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self._load_model(self.weight_dir)
        if self._step == 0:
            self._summary_writer = tf.summary.FileWriter(self.log_dir, self._sess.graph)
        else:
            self._summary_writer = tf.summary.FileWriter(self.log_dir)

    def _discriminator_model(self, input_tensor, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            if set_exists(self.d_arch, 'selu'):
                def forward(inputs, name, filters, use_bn=True, use_dropout=True):
                    inputs = conv2d(inputs, filters, name=name + '/conv2d')
                    if use_bn:
                        inputs = tf.nn.selu(inputs, name=name + '/selu')
                    if use_dropout:
                        inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    return inputs
            elif set_exists(self.d_arch, 'bn_first'):
                def forward(inputs, name, filters, use_bn=True, use_dropout=True):
                    inputs = conv2d(inputs, filters, name=name + '/conv2d')
                    if use_bn:
                        inputs = batch_norm(inputs, name=name + '/bn')
                    inputs = leaky_relu(inputs, name=name + '/lrelu')
                    if use_dropout:
                        inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    return inputs
            elif set_exists(self.d_arch, 'bn_last'):
                def forward(inputs, name, filters, use_bn=True, use_dropout=True):
                    inputs = conv2d(inputs, filters, name=name + '/conv2d')
                    inputs = leaky_relu(inputs, name=name + '/lrelu')
                    if use_dropout:
                        inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    if use_bn:
                        inputs = batch_norm(inputs, name=name + '/bn')
                    return inputs
            elif set_exists(self.d_arch, 'resnet'):
                def forward(inputs, name, filters):
                    input_a = batch_norm(inputs, name=name + '/bn_1')
                    input_a = tf.nn.relu(input_a, name=name + '/relu_1')
                    input_a = conv2d(input_a, input_a.get_shape().as_list()[-1], kernel_size=(3, 3),
                                     strides=(1, 1), name=name + '/conv2d_1')
                    input_a = batch_norm(input_a, name=name + '/bn_2')
                    input_a = tf.nn.relu(input_a, name=name + '/relu_2')
                    input_a = conv2d(input_a, filters, kernel_size=(3, 3), strides=(1, 1), name=name + '/conv2d_2')
                    input_a = tf.nn.avg_pool(input_a, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name=name + '/avgpool')
                    input_b = tf.nn.avg_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name=name + '/avgpool_sc')
                    if filters != input_b.get_shape().as_list()[-1]:
                        input_b = conv2d(input_b, filters, kernel_size=(1, 1), strides=(1, 1), name=name + '/conv2d_sc')
                    return input_a + input_b
            else:
                raise ValueError('Incorrect d_arch')

            # layer 1, not applying bn, uses leaky relu only
            if set_exists(self.d_arch, 'resnet'):
                model = conv2d(input_tensor, self.d_filter, (3, 3), (1, 1), name='layer1/conv2d')
            else:
                model = forward(input_tensor, name='layer1', filters=self.d_filter, use_bn=False, use_dropout=False)
            # layer 2 to 4
            model = forward(model, name='layer2', filters=self.d_filter * 2)
            model = forward(model, name='layer3', filters=self.d_filter * 4)
            model = forward(model, name='layer4', filters=self.d_filter * 8)

            if set_exists(self.d_arch, 'resnet'):
                model = forward(model, name='layer6', filters=self.d_filter * 8)

            model = tf.reshape(model, [tf.shape(model)[0], np.prod(model.get_shape().as_list()[1:])],
                               name='layer5/flatten')
            model_logits = dense(model, 1, name='layer5/dense')
            # model = tf.nn.sigmoid(model_logits, name='layer5/sigmoid')

            return model_logits

    def _generator_model(self, reuse=False, train=True):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            if set_exists(self.g_arch, 'selu'):
                def forward(inputs, output_shape, name, use_deconv=True):
                    if use_deconv:
                        inputs = deconv2d(inputs, output_shape, name=name + '/deconv2d')
                    inputs = tf.nn.selu(inputs, name=name + '/selu')
                    inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    return inputs
            elif set_exists(self.g_arch, 'bn_first'):
                def forward(inputs, output_shape, name, use_deconv=True):
                    if use_deconv:
                        inputs = deconv2d(inputs, output_shape, name=name + '/deconv2d')
                    inputs = batch_norm(inputs, train=train, name=name + '/bn')
                    inputs = tf.nn.relu(inputs, name=name + '/relu')
                    inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    return inputs
            elif set_exists(self.g_arch, 'bn_last'):
                def forward(inputs, output_shape, name, use_deconv=True):
                    if use_deconv:
                        inputs = deconv2d(inputs, output_shape, name=name + '/deconv2d')
                    inputs = tf.nn.relu(inputs, name=name + '/relu')
                    inputs = tf.nn.dropout(inputs, self._dropout_tensor, name=name + '/dropout')
                    inputs = batch_norm(inputs, train=train, name=name + '/bn')
                    return inputs
            elif set_exists(self.g_arch, 'resnet'):
                def forward(inputs, output_shape, name, use_deconv=False):
                    _ = use_deconv  # ignore this param
                    input_a = batch_norm(inputs, train=train, name=name + '/bn_1')
                    input_a = tf.nn.relu(input_a, name=name + '/relu_1')
                    input_a = tf.concat([input_a] * 4, axis=3, name=name + '/upsampling/concat')
                    input_a = tf.depth_to_space(input_a, 2, name=name + '/upsampling/depth_to_space')
                    input_a = conv2d(input_a, output_shape[-1], kernel_size=(3, 3),
                                     strides=(1, 1), name=name + '/conv2d_1')
                    input_a = batch_norm(input_a, train=train, name=name + '/bn_2')
                    input_a = tf.nn.relu(input_a, name=name + '/relu_2')
                    input_a = conv2d(input_a, output_shape[-1], kernel_size=(3, 3),
                                     strides=(1, 1), name=name + '/conv2d_2')
                    input_b = tf.concat([inputs] * 4, axis=3, name=name + '/upsampling/concat')
                    input_b = tf.depth_to_space(input_b, 2, name=name + '/upsampling/depth_to_space')
                    if output_shape[-1] != input_b.get_shape().as_list()[-1]:
                        input_b = conv2d(input_b, output_shape[-1], kernel_size=(1, 1),
                                         strides=(1, 1), name=name + '/conv2d_sc')
                    output = input_a + input_b
                    actual_output_shape = output.get_shape().as_list()
                    if actual_output_shape[1:3] != output_shape[1:3]:
                        offset_h = (actual_output_shape[1] - output_shape[1]) // 2
                        offset_w = (actual_output_shape[2] - output_shape[2]) // 2
                        assert offset_h >= 0 and offset_w >= 0
                        output = output[:, offset_h: offset_h+output_shape[1], offset_w: offset_w+output_shape[2], :]
                    return output
            else:
                raise ValueError('Incorrect g_arch')

            # fc layer
            size_h = calc_conv_out_shape_same(self.image_height, 16)
            size_w = calc_conv_out_shape_same(self.image_width, 16)
            model = dense(self._noise_tensor, size_h * size_w * self.g_filter * 8, name='layer0/fc')
            model = tf.reshape(model, [-1, size_h, size_w, self.g_filter * 8], name='layer0/reshape')

            if not set_exists(self.g_arch, 'resnet'):
                model = forward(model, [None, size_h, size_w, self.g_filter * 8], name='layer0', use_deconv=False)

            # deconv1
            size_h = calc_conv_out_shape_same(self.image_height, 8)
            size_w = calc_conv_out_shape_same(self.image_width, 8)
            if not set_exists(self.g_arch, 'resnet'):
                model = forward(model, [None, size_h, size_w, self.g_filter * 4], name='layer1')
            else:
                model = forward(model, [None, size_h, size_w, self.g_filter * 8], name='layer0')

            # deconv2
            size_h = calc_conv_out_shape_same(self.image_height, 4)
            size_w = calc_conv_out_shape_same(self.image_width, 4)
            if not set_exists(self.g_arch, 'resnet'):
                model = forward(model, [None, size_h, size_w, self.g_filter * 2], name='layer2')
            else:
                model = forward(model, [None, size_h, size_w, self.g_filter * 4], name='layer1')

            # deconv3
            size_h = calc_conv_out_shape_same(self.image_height, 2)
            size_w = calc_conv_out_shape_same(self.image_width, 2)
            if not set_exists(self.g_arch, 'resnet'):
                model = forward(model, [None, size_h, size_w, self.g_filter], name='layer3')
            else:
                model = forward(model, [None, size_h, size_w, self.g_filter * 2], name='layer2')

            # deconv4(output layer)
            if not set_exists(self.g_arch, 'resnet'):
                model = deconv2d(model, [None, self.image_height, self.image_width, self.channel_count],
                                 name='layer4/deconv2d')
            else:
                model = forward(model, [None, self.image_height, self.image_width, self.g_filter], name='layer3')
                model = batch_norm(model, train=train, name='layer4/bn')
                model = tf.nn.relu(model, name='layer4/relu')
                model = conv2d(model, self.channel_count, (3, 3), (1, 1), name='layer4/conv2d')
            model = tf.nn.tanh(model, name='layer_output/tanh')

            return model

    def _compile_loss(self):
        g_loss = -tf.reduce_mean(self._gan_logits)
        d_loss = -tf.reduce_mean(self._d_logits) + tf.reduce_mean(self._gan_logits)
        alpha = tf.random_uniform(shape=[tf.shape(self._d_logits)[0], 1, 1, 1], minval=0, maxval=1)
        interpolates = alpha * self._input_tensor + (1 - alpha) * self._g
        d_interpolates = self._discriminator_model(interpolates, True)
        gradient = tf.gradients(d_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradient), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1) ** 2)
        self._summary_dict['gp'] = tf.summary.scalar('gradient_penalty', gradient_penalty)
        d_loss += 10 * gradient_penalty
        self._summary_dict['d_loss'] = tf.summary.scalar('d_loss', d_loss)  # test summary
        self._summary_dict['g_loss'] = tf.summary.scalar('g_loss', g_loss)
        return d_loss, g_loss

    # this code is for saving the current weights of discriminator, generator model
    def _save_model(self, fname):
        self._saver.save(self._sess, fname + '/gan', global_step=self._step)
        print('\nmodel saved, save step %d' % self._step)

    # this code is for loading the saved weights of discriminator, generator model
    def _load_model(self, fname):
        ckpt = tf.train.get_checkpoint_state(fname)
        if ckpt and ckpt.model_checkpoint_path:
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)
            self._step = int(re.search(re.compile('\\d+$'), ckpt.model_checkpoint_path)[0])
        print('\nmodel loaded, restore step %d' % self._step)

    def train_model(self):
        # samples of training set
        m = self._async_io.get_sample_count()

        # the epoch counter (from 0 every time)
        i = 0
        # the counter for d_opt runs
        d_opt_has_ran = 0

        if self.epochs <= 0:
            stop_interrupter = AsyncInput()
            print('Press "Enter" to exit')
        else:
            stop_interrupter = None

        while True:
            i += 1
            if self.epochs <= 0:
                # only works in windows with console window
                if stop_interrupter.is_stop():
                    break
                print('[Epoch %d]' % i)
            else:
                if i > self.epochs:
                    break
                print('[Epoch %d of %d]' % (i, self.epochs))

            # how many steps should be run per epoch
            steps = int(math.ceil(m / self.batch_size))
            for _ in tqdm(range(steps), ascii=True):
                if stop_interrupter.is_stop():
                    break
                self._step += 1
                # summarize the learning rate
                summary, summary2 = self._sess.run([self._summary_dict['d_lr'], self._summary_dict['d_beta1']])
                self._summary_writer.add_summary(summary, self._step)
                self._summary_writer.add_summary(summary2, self._step)
                summary, summary2 = self._sess.run([self._summary_dict['g_lr'], self._summary_dict['g_beta1']])
                self._summary_writer.add_summary(summary, self._step)
                self._summary_writer.add_summary(summary2, self._step)

                # the sample length of current step
                images_real = self._async_io.wait_io()
                length = images_real.shape[0]
                noise = np.random.uniform(-1.0, 1.0, size=[length, self.noise_dim])

                # training the discriminator
                _, summary, summary2, summary3 = self._sess.run(
                    [self._d_opt, self._summary_dict['d_loss'],
                     self._summary_dict['d_pred'], self._summary_dict['gp']],
                    feed_dict={self._input_tensor: images_real, self._noise_tensor: noise,
                               self._dropout_tensor: 1.0 - self.dropout_rate})
                self._summary_writer.add_summary(summary, self._step)
                self._summary_writer.add_summary(summary2, self._step)
                self._summary_writer.add_summary(summary3, self._step)
                d_opt_has_ran += 1
                # continue to train the discriminator if d_opt_runs_per_step > 1 (skips training the generator)
                if d_opt_has_ran < self.d_opt_runs_per_step:
                    self._step -= 1
                    continue
                d_opt_has_ran %= self.d_opt_runs_per_step

                for _ in range(self.g_opt_runs_per_step):
                    _, summary = self._sess.run(
                        [self._g_opt, self._summary_dict['g_loss']],
                        feed_dict={self._noise_tensor: noise, self._dropout_tensor: 1.0 - self.dropout_rate})
                    self._summary_writer.add_summary(summary, self._step)

                # testing generator
                if self._step % self.test_generator_per_step == 0:
                    noise = np.random.uniform(-1.0, 1.0, size=[self.random_count, self.noise_dim])
                    img_pred, summary = self._sess.run(
                        [self.sampler, self._summary_dict['g_pred']],
                        feed_dict={self._noise_tensor: noise, self._dropout_tensor: 1.0})
                    self._summary_writer.add_summary(summary, self._step)
                    img_trace, summary = self._sess.run(
                        [self.sampler, self._summary_dict['g_trace']],
                        feed_dict={self._noise_tensor: self._sample_noise, self._dropout_tensor: 1.0})
                    self._summary_writer.add_summary(summary, self._step)

                    save_images(os.path.join(self.output_dir, 'pred_%d_steps.png' % self._step),
                                rescale_to_rgb(img_pred))
                    save_images(os.path.join(self.output_dir, 'trace_%d_steps.png' % self._step),
                                rescale_to_rgb(img_trace))
                # saving weights
                if self._step % self.save_model_per_step == 0:
                    self._save_model(self.weight_dir)

                # end step for
            # end epoch for
        # save model after exiting training process
        self._save_model(self.weight_dir)


def main():
    # defining the output image shape for the generator
    image_width = 100
    image_height = 100
    # defining the dimension of noise vector for the generator (also called z_dim)
    noise_dim = 100
    # defining the minimal filter size for g(generator) and d(discriminator)
    # for the DCGAN paper, the filter size is [1024, 512, 256, 128] (except the last value 3), so set this value to 128
    g_filter = 64
    d_filter = 64
    # defining the batch size for batch training
    batch_size = 64
    # defining the file path for loading the training set
    training_set_path = 'animeface-np'
    # defining the image color channel count (for default only, will be overwritten after loading the training set)
    channel_count = 3
    # defining the learning rate and momentum value for Adam optimizer
    d_adam_lr = 0.0001
    d_adam_beta1 = 0.5
    g_adam_lr = 0.0001
    g_adam_beta1 = 0.5
    # defining the number used to generate the image during the training process
    # `random_count` is used to generate images with random noise
    # `sample_count` is used to generate images with fixed noise
    random_count = 64
    sample_count = 64
    # defining the dropout rate for the learning process, set it to 0.0 to disable dropout layer
    dropout_rate = 0.2
    # defining how many training steps the generator should be tested
    test_generator_per_step = 100
    # defining how many training steps the weights should be saved
    save_weights_per_step = 500
    # defining how many times discriminator and generator optimizer should run in one training step
    d_opt_runs_per_step = 5
    g_opt_runs_per_step = 1
    # defining the directory for storing model weights, the logs and generator outputs
    weight_dir = 'model.test1'
    log_dir = 'log.test1'
    output_dir = 'output.test1'
    # defining the architecture of D and G
    # `selu` uses the SeLU activation function (Self-normalized Linear Unit), only followed by a dropout layer
    # `bn_first` uses ReLU activation for G and LeakyReLU for D, ordered by (De)Conv2D -> BN -> Activation -> Dropout
    # `bn_last` uses ReLU activation for G and LeakyReLU for G, ordered by (De)Conv2D -> Activation -> Dropout -> BN
    # `resnet` uses residual block instead of a normal conv. block, performing jump connection
    d_arch = {'bn_first'}
    g_arch = {'bn_first'}
    # epochs for training, set it to -1 if you want to exit the program by pressing `Enter`
    epochs = -1

    model = WGAN(training_set_path, batch_size, image_width, image_height, channel_count, d_filter, g_filter, epochs,
                 d_arch, g_arch, noise_dim, d_adam_lr, d_adam_beta1, g_adam_lr, g_adam_beta1, dropout_rate,
                 test_generator_per_step, save_weights_per_step, weight_dir, log_dir, output_dir, sample_count,
                 random_count, d_opt_runs_per_step, g_opt_runs_per_step)
    model.train_model()


if __name__ == '__main__':
    main()
