import os
import math
import re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io
from getch import getch


# a class to store variables in a dictionary
class Vars(object):
    def __init__(self):
        self.d = dict()

    def get(self, key):
        return self.d[key]

    def set(self, key, value):
        self.d[key] = value

    def __iter__(self):
        for x in self.d:
            yield x


global_var = Vars()


def load_training_set():
    input_path = global_var.get('training_set_path')
    cache_path = os.path.join(input_path, '../train.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)
    files = os.listdir(input_path)
    list_images = []
    for file in files:
        abs_path = os.path.join(input_path, file)
        if os.path.isfile(abs_path):
            image = io.imread(abs_path)
            list_images.append(image)
    ret_tensor = np.array(list_images)
    # rescaling data
    ret_tensor = (ret_tensor - 127.5) / 127.5
    ret_tensor = ret_tensor.astype(np.float32)
    np.save(cache_path, ret_tensor)
    return ret_tensor


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


def dense(i, output_dim, name='linear', stddev=0.02, output_weights=False):
    shape = i.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
        mul = tf.matmul(i, w) + b
    if output_weights:
        return mul, w, b
    else:
        return mul


def batch_norm(i, epsilon=1e-5, momentum=0.9, train=True, name='batch_norm'):
    return tf.contrib.layers.batch_norm(i, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True,
                                        is_training=train, scope=name)


def calc_conv_out_shape_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def discriminator_model(input_tensor, dropout_tensor, reuse=False):
    d_filter = global_var.get('d_filter')
    d_arch = global_var.get('d_arch')
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        # layer 1, not applying bn, uses leaky relu only
        model = conv2d(input_tensor, d_filter, name='layer1/conv2d')
        model = leaky_relu(model, name='layer1/lrelu')

        if d_arch == 'selu':
            def forward(inputs, name):
                inputs = tf.nn.selu(inputs, name=name + '/selu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                return inputs
        elif d_arch == 'bn_first':
            def forward(inputs, name):
                inputs = batch_norm(inputs, name=name + '/bn')
                inputs = leaky_relu(inputs, name=name + '/lrelu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                return inputs
        elif d_arch == 'bn_last':
            def forward(inputs, name):
                inputs = leaky_relu(inputs, name=name + '/lrelu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                inputs = batch_norm(inputs, name=name + '/bn')
                return inputs
        else:
            raise ValueError('Incorrect d_arch')

        # layer 2 to 4
        model = conv2d(model, d_filter * 2, name='layer2/conv2d')
        model = forward(model, name='layer2')

        model = conv2d(model, d_filter * 4, name='layer3/conv2d')
        model = forward(model, name='layer3')

        model = conv2d(model, d_filter * 8, name='layer4/conv2d')
        model = forward(model, name='layer4')

        model = tf.reshape(model, [tf.shape(model)[0], np.prod(model.get_shape().as_list()[1:])],
                           name='layer5/flatten')
        model_logits = dense(model, 1, name='layer5/dense')
        model = tf.nn.sigmoid(model_logits, name='layer5/sigmoid')

        return model, model_logits


def generator_model(input_tensor, dropout_tensor, reuse=False, train=True):
    image_width = global_var.get('image_width')
    image_height = global_var.get('image_height')
    g_filter = global_var.get('g_filter')
    channel_count = global_var.get('channel_count')
    g_arch = global_var.get('g_arch')
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()

        if g_arch == 'selu':
            def forward(inputs,  name):
                inputs = tf.nn.selu(inputs, name=name + '/selu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                return inputs
        elif g_arch == 'bn_first':
            def forward(inputs, name):
                inputs = batch_norm(inputs, train=train, name=name + '/bn')
                inputs = tf.nn.relu(inputs, name=name + '/relu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                return inputs
        elif g_arch == 'bn_last':
            def forward(inputs, name):
                inputs = tf.nn.relu(inputs, name=name + '/relu')
                inputs = tf.nn.dropout(inputs, dropout_tensor, name=name + '/dropout')
                inputs = batch_norm(inputs, train=train, name=name + '/bn')
                return inputs
        else:
            raise ValueError('Incorrect g_arch')

        # fc layer
        size_h = calc_conv_out_shape_same(image_height, 16)
        size_w = calc_conv_out_shape_same(image_width, 16)
        model = dense(input_tensor, size_h * size_w * g_filter * 8, name='layer0/fc')
        model = tf.reshape(model, [-1, size_h, size_w, g_filter * 8], name='layer0/reshape')
        model = forward(model, name='layer0')

        # deconv1
        size_h = calc_conv_out_shape_same(image_height, 8)
        size_w = calc_conv_out_shape_same(image_width, 8)
        model = deconv2d(model, [None, size_h, size_w, g_filter * 4], name='layer1/deconv2d')
        model = forward(model, name='layer1')

        # deconv2
        size_h = calc_conv_out_shape_same(image_height, 4)
        size_w = calc_conv_out_shape_same(image_width, 4)
        model = deconv2d(model, [None, size_h, size_w, g_filter * 2], name='layer2/deconv2d')
        model = forward(model, name='layer2')

        # deconv3
        size_h = calc_conv_out_shape_same(image_height, 2)
        size_w = calc_conv_out_shape_same(image_width, 2)
        model = deconv2d(model, [None, size_h, size_w, g_filter], name='layer3/deconv2d')
        model = forward(model, name='layer3')

        # deconv4(output layer)
        model = deconv2d(model, [None, image_height, image_width, channel_count], name='layer4/deconv2d')
        model = tf.nn.tanh(model, name='layer4/tanh')

        return model


def compile_loss(d_logits, gan_logits, summary_dict):
    d_loss_true = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.ones_like(d_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits, labels=tf.zeros_like(gan_logits)))
    d_loss = d_loss_true + d_loss_fake

    summary_dict['d_loss'] = tf.summary.scalar('d_loss', d_loss)  # test summary
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gan_logits, labels=tf.ones_like(gan_logits)))

    summary_dict['g_loss'] = tf.summary.scalar('g_loss', g_loss)
    return d_loss, g_loss


prev_step = 0


# this code is for saving the current weights of discriminator, generator model
def save_model(sess, saver, fname, step):
    saver.save(sess, fname + '/gan', global_step=step)
    print('\nmodel saved, save step %d' % step)
    return saver


# this code is for loading the saved weights of discriminator, generator model
def load_model(sess, saver, fname):
    ckpt = tf.train.get_checkpoint_state(fname)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global prev_step
        prev_step = int(re.search(re.compile('\\d+$'), ckpt.model_checkpoint_path)[0])
    print('\nmodel loaded, restore step %d' % prev_step)
    return saver


def save_images(output_name, images):
    m, h, w, c = images.shape
    rows = int(math.ceil(math.sqrt(m)))
    cols = rows
    out_image = np.zeros((rows * h, cols * w, c))
    for y in range(rows):
        for x in range(cols):
            offset = y * cols + x
            if offset >= m:
                continue
            out_image[y*h:(y+1)*h, x*w:(x+1)*w, :] = images[offset]
    io.imsave(output_name, out_image)


def train_model(sess, saver, train_x,
                d_opt, g_opt, sampler,
                image_input, noise_input, sample_noise, dropout_input,
                summary_dict, summary_writer):
    # retrieving the global variables
    m = train_x.shape[0]
    batch_size = global_var.get('batch_size')
    noise_dim = global_var.get('noise_dim')
    dropout_rate = global_var.get('dropout_rate')
    test_generator_per_step = global_var.get('test_generator_per_step')
    save_weights_per_step = global_var.get('save_weights_per_step')
    random_count = global_var.get('random_count')
    d_opt_runs_per_step = global_var.get('d_opt_runs_per_step')
    g_opt_runs_per_step = global_var.get('g_opt_runs_per_step')
    epochs = global_var.get('epochs')

    # updating the step counter
    global prev_step
    step_sum = prev_step
    # retrieving the summary variables
    d_loss_sum = summary_dict['d_loss']
    g_loss_sum = summary_dict['g_loss']

    d_pred = summary_dict['d_pred']
    g_pred = summary_dict['g_pred']
    g_trace = summary_dict['g_trace']
    d_lr_sum = summary_dict['d_lr']
    d_beta1_sum = summary_dict['d_beta1']
    g_lr_sum = summary_dict['g_lr']
    g_beta1_sum = summary_dict['g_beta1']
    # the epoch counter (from 0 every time)
    i = 0
    # the counter for d_opt runs
    d_opt_has_ran = 0
    
    while True:
        i += 1
        if epochs <= 0:
            if getch() == 'q':
                break
            print('[Epoch %d]' % i)
        else:
            if i > epochs:
                break
            print('[Epoch %d of %d]' % (i, epochs))

        # randomize the indices for the training set
        random_idx = np.arange(m)
        np.random.shuffle(random_idx)
        # calculating how many steps should be run for one epoch
        steps = int(math.ceil(m / batch_size))
        for step in tqdm(range(steps), ascii=True):
            # summarize the learning rate
            summary, summary2 = sess.run([d_lr_sum, d_beta1_sum])
            summary_writer.add_summary(summary, step_sum)
            summary_writer.add_summary(summary2, step_sum)
            summary, summary2 = sess.run([g_lr_sum, g_beta1_sum])
            summary_writer.add_summary(summary, step_sum)
            summary_writer.add_summary(summary2, step_sum)

            # the indices of training set for current step
            step_idx = random_idx[step * batch_size: (step + 1) * batch_size]
            # the sample length of current step
            length = len(step_idx)
            images_real = train_x[step_idx]
            noise = np.random.uniform(-1.0, 1.0, size=[length, noise_dim])

            # training the discriminator
            _, summary, summary2 = sess.run([d_opt, d_loss_sum, d_pred], feed_dict={image_input: images_real,
                                                                                    noise_input: noise,
                                                                                    dropout_input: 1.0 - dropout_rate})
            summary_writer.add_summary(summary, step_sum)
            summary_writer.add_summary(summary2, step_sum)
            d_opt_has_ran += 1
            # continue to train the discriminator if d_opt_runs_per_step > 1 (skips training the generator)
            if d_opt_has_ran < d_opt_runs_per_step:
                continue
            d_opt_has_ran %= d_opt_runs_per_step

            for _ in range(g_opt_runs_per_step):
                _, summary = sess.run([g_opt, g_loss_sum], feed_dict={noise_input: noise,
                                                                      dropout_input: 1.0 - dropout_rate})
                summary_writer.add_summary(summary, step_sum)

            # testing generator
            if (step_sum + 1) % test_generator_per_step == 0:
                noise = np.random.uniform(-1.0, 1.0, size=[random_count, noise_dim])
                img_pred, summary = sess.run([sampler, g_pred], feed_dict={noise_input: noise, dropout_input: 1.0})
                summary_writer.add_summary(summary, step_sum + 1)
                img_trace, summary = sess.run([sampler, g_trace], feed_dict={noise_input: sample_noise,
                                                                             dropout_input: 1.0})
                summary_writer.add_summary(summary, step_sum + 1)
                
                save_images(global_var.get('output_dir') + '/pred_%d_steps.png' % (step_sum + 1),
                            rescale_to_rgb(img_pred))
                save_images(global_var.get('output_dir') + '/trace_%d_steps.png' % (step_sum + 1),
                            rescale_to_rgb(img_trace))
            # saving weights
            if (step_sum + 1) % save_weights_per_step == 0:
                saver = save_model(sess, saver, global_var.get('weight_dir'), step_sum + 1)

            step_sum += 1
            # end step for
        # end epoch for
    # save model after exiting training process
    save_model(sess, saver, global_var.get('weight_dir'), step_sum)
    prev_step = step_sum


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
    batch_size = 128
    # defining the file path for loading the training set
    training_set_path = 'train'
    # defining the image color channel count (for default only, will be overwritten after loading the training set)
    channel_count = 3
    # defining the learning rate and momentum value for Adam optimizer
    d_adam_lr = 0.0002
    d_adam_beta1 = 0.5
    g_adam_lr = 0.0002
    g_adam_beta1 = 0.5
    # defining the number used to generate the image during the training process
    # `random_count` is used to generate images with random noise
    # `sample_count` is used to generate images with fixed noise
    random_count = 64
    sample_count = 64
    # defining the dropout rate for the learning process, set it to 0.0 to disable dropout layer
    dropout_rate = 0.0
    # defining how many training steps the generator should be tested
    test_generator_per_step = 100
    # defining how many training steps the weights should be saved
    save_weights_per_step = 500
    # defining how many times discriminator and generator optimizer should run in one training step
    d_opt_runs_per_step = 1
    g_opt_runs_per_step = 1
    # defining the directory for storing model weights, the logs and generator outputs
    weight_dir = 'model.run1'
    log_dir = 'log.run1'
    output_dir = 'output.run1'
    # defining the architecture of D and G
    # `selu` uses the SeLU activation function (Self-normalized Linear Unit), only followed by a dropout layer
    # `bn_first` uses ReLU activation for G and LeakyReLU for D, ordered by (De)Conv2D -> BN -> Activation -> Dropout
    # `bn_last` uses ReLU activation for G and LeakyReLU for G, ordered by (De)Conv2D -> Activation -> Dropout -> BN
    d_arch = 'bn_first'
    g_arch = 'bn_first'
    # epochs for training, set it to -1 if you want to exit the program by pressing `Q`
    epochs = -1

    # save to global variables
    global_var.set('image_width', image_width)
    global_var.set('image_height', image_height)
    global_var.set('noise_dim', noise_dim)
    global_var.set('g_filter', g_filter)
    global_var.set('d_filter', d_filter)
    global_var.set('batch_size', batch_size)
    global_var.set('training_set_path', training_set_path)
    global_var.set('channel_count', channel_count)
    global_var.set('d_adam_lr', d_adam_lr)
    global_var.set('d_adam_beta1', d_adam_beta1)
    global_var.set('g_adam_lr', g_adam_lr)
    global_var.set('g_adam_beta1', g_adam_beta1)
    global_var.set('random_count', random_count)
    global_var.set('sample_count', sample_count)
    global_var.set('dropout_rate', dropout_rate)
    global_var.set('test_generator_per_step', test_generator_per_step)
    global_var.set('save_weights_per_step', save_weights_per_step)
    global_var.set('d_opt_runs_per_step', d_opt_runs_per_step)
    global_var.set('g_opt_runs_per_step', g_opt_runs_per_step)
    global_var.set('weight_dir', weight_dir)
    global_var.set('log_dir', log_dir)
    global_var.set('output_dir', output_dir)
    global_var.set('d_arch', d_arch)
    global_var.set('g_arch', g_arch)
    global_var.set('epochs', epochs)

    # validation test for architecture string
    if d_arch not in ['selu', 'bn_first', 'bn_last']:
        raise ValueError('d_arch should be one of "selu", "bn_first" or "bn_last"')
    if g_arch not in ['selu', 'bn_first', 'bn_last']:
        raise ValueError('g_arch should be one of "selu", "bn_first" or "bn_last"')

    # loading the training set into train_x, updating the channel_count
    print('** LOADING TRAINING SET **')
    train_x = load_training_set()
    channel_count = train_x.shape[-1]
    global_var.set('channel_count', channel_count)
    print(train_x.shape, train_x.dtype)

    # constructing the graph
    print('** CONSTRUCTING VARIABLES AND COMPUTE GRAPH **')
    image_input = tf.placeholder(tf.float32, [None, image_height, image_width, channel_count],
                                 name='discriminator/input')
    noise_input = tf.placeholder(tf.float32, [None, noise_dim], name='generator/input')
    dropout_input = tf.placeholder(tf.float32, name='dropout_rate')
    summary_dict = dict()

    d, d_logits = discriminator_model(image_input, dropout_input)
    g = generator_model(noise_input, dropout_input)
    gan, gan_logits = discriminator_model(g, dropout_input, reuse=True)
    sampler = generator_model(noise_input, dropout_input, reuse=True, train=False)

    # generating the loss function
    d_loss, g_loss = compile_loss(d_logits, gan_logits, summary_dict)
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]
    d_opt = tf.train.AdamOptimizer(d_adam_lr, beta1=d_adam_beta1).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(g_adam_lr, beta1=g_adam_beta1).minimize(g_loss, var_list=g_vars)

    tf.contrib.slim.model_analyzer.analyze_vars(t_vars, print_info=True)

    # set the sample noise using the specified seed
    np.random.seed(0)
    sample_noise = np.random.uniform(-1.0, 1.0, size=[sample_count, noise_dim])
    import time
    t = int(time.time())
    np.random.seed(t)

    summary_dict['d_pred'] = tf.summary.histogram('d_pred', tf.reshape(tf.concat([d, gan], axis=0), [1, -1]))
    summary_dict['g_pred'] = tf.summary.image('g_pred', g, max_outputs=random_count)
    summary_dict['g_trace'] = tf.summary.image('g_trace', g, max_outputs=sample_count)
    summary_dict['d_lr'] = tf.summary.scalar('d_lr', d_adam_lr)
    summary_dict['d_beta1'] = tf.summary.scalar('d_beta1', d_adam_beta1)
    summary_dict['g_lr'] = tf.summary.scalar('g_lr', g_adam_lr)
    summary_dict['g_beta1'] = tf.summary.scalar('g_beta1', g_adam_beta1)

    # starting the session
    print("** STARTING SESSION **")
    saver = tf.train.Saver()
    sess = set_vram_growth()
    tf.global_variables_initializer().run(session=sess)
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    saver = load_model(sess, saver, weight_dir)

    # creating new directory
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # start training process
    print('** TRAINING PROCESS STARTED **')
    if epochs <= 0:
        print('Press "Q" to exit this program')

    train_model(sess, saver, train_x, d_opt, g_opt, sampler, image_input, noise_input, sample_noise, dropout_input,
                summary_dict, summary_writer)

    sess.close()
    summary_writer.close()


if __name__ == '__main__':
    main()
