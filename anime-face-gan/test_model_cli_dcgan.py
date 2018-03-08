import os, random, math, re
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io
from getch import getch


def load_training_set(input_path):
    # cache numpy array
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
    return tf.maximum(x, alpha * x)


def dense(i, output_dim, name=None, stddev=0.02, output_weights=False):
    shape = i.get_shape().as_list()
    with tf.variable_scope(name or 'linear'):
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


def discriminator_model(input_tensor, d_filter, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        model = conv2d(input_tensor, d_filter, name='layer1/conv2d')
        model = leaky_relu(model, name='layer1/lrelu')

        model = conv2d(model, d_filter * 2, name='layer2/conv2d')
        model = batch_norm(model, name='layer2/bn')
        model = leaky_relu(model, name='layer2/lrelu')

        model = conv2d(model, d_filter * 4, name='layer3/conv2d')
        model = batch_norm(model, name='layer3/bn')
        model = leaky_relu(model, name='layer3/lrelu')

        model = conv2d(model, d_filter * 8, name='layer4/conv2d')
        model = batch_norm(model, name='layer4/bn')
        model = leaky_relu(model, name='layer4/lrelu')

        model = tf.reshape(model, [tf.shape(model)[0], np.prod(model.get_shape().as_list()[1:])], name='layer5/flatten')
        model_logits = dense(model, 1, name='layer5/dense')
        model = tf.nn.sigmoid(model_logits, name='layer5/sigmoid')

        return model, model_logits


def generator_model(input_tensor, image_width, image_height, g_filter, channel_count, reuse=False, train=True):
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()

        # fc layer
        size_h = calc_conv_out_shape_same(image_height, 16)
        size_w = calc_conv_out_shape_same(image_width, 16)
        model = dense(input_tensor, size_h * size_w * g_filter * 8, name='layer0/fc')
        model = tf.reshape(model, [-1, size_h, size_w, g_filter * 8], name='layer0/reshape')
        model = batch_norm(model, name='layer0/bn', train=train)
        model = tf.nn.relu(model, name='layer0/relu')

        # deconv1
        size_h = calc_conv_out_shape_same(image_height, 8)
        size_w = calc_conv_out_shape_same(image_width, 8)
        model = deconv2d(model, [None, size_h, size_w, g_filter * 4], name='layer1/deconv2d')
        model = batch_norm(model, name='layer1/bn', train=train)
        model = tf.nn.relu(model, name='layer1/relu')

        # deconv2
        size_h = calc_conv_out_shape_same(image_height, 4)
        size_w = calc_conv_out_shape_same(image_width, 4)
        model = deconv2d(model, [None, size_h, size_w, g_filter * 2], name='layer2/deconv2d')
        model = batch_norm(model, name='layer2/bn', train=train)
        model = tf.nn.relu(model, name='layer2/relu')

        # deconv3
        size_h = calc_conv_out_shape_same(image_height, 2)
        size_w = calc_conv_out_shape_same(image_width, 2)
        model = deconv2d(model, [None, size_h, size_w, g_filter], name='layer3/deconv2d')
        model = batch_norm(model, name='layer3/bn', train=train)
        model = tf.nn.relu(model, name='layer3/relu')

        # deconv4(output layer)
        model = deconv2d(model, [None, image_height, image_width, channel_count], name='layer4/deconv2d')
        model = tf.nn.tanh(model, name='layer4/tanh')

        return model


def compile_loss(D_logits, GAN_logits):
    d_loss_true = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits, labels=tf.zeros_like(GAN_logits)))
    d_loss = d_loss_true + d_loss_fake

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits, labels=tf.ones_like(GAN_logits)))

    return d_loss, g_loss


prev_step = 0


# this code is for saving the current weights of discriminator, generator model
def save_model(sess, saver, fname, step):
    saver.save(sess, fname + '/gan', global_step=step)
    return saver


# this code is for loading the saved weights of discriminator, generator model
def load_model(sess, saver, fname):
    ckpt = tf.train.get_checkpoint_state(fname)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global prev_step
        prev_step = int(re.search(re.compile('\\d+$'), ckpt.model_checkpoint_path)[0])
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

    
def test_model(sess, sampler, noise_input, sample_count, noise_dim):
    noise = np.random.uniform(-1.0, 1.0, size=[sample_count, noise_dim])
    images = sess.run(sampler, feed_dict={noise_input: noise})
    return images
    
def main():
    image_width = 100
    image_height = 100
    noise_dim = 100
    g_filter = 64
    d_filter = 64
    sample_count = 64
    output_filename = 'output_%d.png' % sample_count
    
    test_generator_per_step = 100
    save_weights_per_step = 500

    print('** CONSTRUCTING VARIABLES AND COMPUTE GRAPH **')
    image_input = tf.placeholder(tf.float32, [None, image_height, image_width, channel_count],
                                 name='discriminator/input')
    noise_input = tf.placeholder(tf.float32, [None, noise_dim], name='generator/input')

    D, D_logits = discriminator_model(image_input, d_filter)
    G = generator_model(noise_input, image_width, image_height, g_filter, channel_count)
    GAN, GAN_logits = discriminator_model(G, d_filter, reuse=True)
    Sampler = generator_model(noise_input, image_width, image_height, g_filter, channel_count, reuse=True, train=False)

    d_loss, g_loss = compile_loss(D_logits, GAN_logits, summary_dict)
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = [var for var in t_vars if 'generator' in var.name]
    d_opt = tf.train.AdamOptimizer(adam_lr, beta1=adam_beta1).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(adam_lr, beta1=adam_beta1).minimize(g_loss, var_list=g_vars)

    np.random.seed(0)
    sample_noise = np.random.uniform(-1.0, 1.0, size=[sample_count, noise_dim])
    import time
    t = int(time.time())
    np.random.seed(t)
    print("** STARTING SESSION **")
    saver = tf.train.Saver()
    sess = set_vram_growth()
    tf.global_variables_initializer().run(session=sess)

    saver = load_model(sess, saver, 'model')

    print("** GENERATING IMAGES **")
    images = test_model(sess, Sampler, noise_input, sample_count, noise_dim)
    save_images(output_filename, images)
    
    sess.close()


if __name__ == '__main__':
    main()
