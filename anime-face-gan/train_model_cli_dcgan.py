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


def compile_loss(D_logits, GAN_logits, summary_dict):
    d_loss_true = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits, labels=tf.zeros_like(GAN_logits)))
    d_loss = d_loss_true + d_loss_fake

    summary_dict['d_loss'] = tf.summary.scalar('d_loss', d_loss)  # test summary
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits, labels=tf.ones_like(GAN_logits)))

    summary_dict['g_loss'] = tf.summary.scalar('g_loss', g_loss)
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

def train_model(sess, saver, train_x,
                noise_dim, random_count,
                d_opt, g_opt, sampler,
                image_input, noise_input, sample_noise,
                summary_dict, summary_writer,
                batch_size=32, test_generator_per_step=100, save_weights_per_step=500):
    m = train_x.shape[0]

    global prev_step
    step_sum = prev_step
    d_loss_sum = summary_dict['d_loss']
    g_loss_sum = summary_dict['g_loss']

    d_pred = summary_dict['d_pred']
    g_pred = summary_dict['g_pred']
    g_trace = summary_dict['g_trace']
    i = 0
    
    while True:
        if getch() == 'q':
            break
        i += 1
        print('[Epoch %d]' % i)

        # 随机顺序
        random_idx = np.arange(m)
        np.random.shuffle(random_idx)
        # 对每个样本进行batch迭代
        steps = int(math.ceil(m / batch_size))
        for step in tqdm(range(steps), ascii=True):
            # 获取当前迭代的随机训练数据
            step_idx = random_idx[step * batch_size: (step + 1) * batch_size]
            length = len(step_idx)
            images_real = train_x[step_idx]
            noise = np.random.uniform(-1.0, 1.0, size=[length, noise_dim])

            _, summary, summary2 = sess.run([d_opt, d_loss_sum, d_pred],
                                            feed_dict={image_input: images_real, noise_input: noise})
            summary_writer.add_summary(summary, step_sum)
            summary_writer.add_summary(summary2, step_sum)

            # run twice g_opt
            sess.run([g_opt], feed_dict={noise_input: noise})
            _, summary = sess.run([g_opt, g_loss_sum], feed_dict={noise_input: noise})
            summary_writer.add_summary(summary, step_sum)

            # run third g_opt (with different noise)
            # for this training process, the discriminator loss is actually decreasing while generator loss does not change
            
            #noise = np.random.uniform(-1.0, 1.0, size=[length, noise_dim])
            #sess.run([g_opt], feed_dict={noise_input: noise})
            #_, summary = sess.run([g_opt, g_loss_sum], feed_dict={noise_input: noise})
            #summary_writer.add_summary(summary, step_sum)
            
            # testing G
            if (step_sum + 1) % test_generator_per_step == 0:
                noise = np.random.uniform(-1.0, 1.0, size=[random_count, noise_dim])
                img_pred, summary = sess.run([sampler, g_pred], feed_dict={noise_input: noise})
                summary_writer.add_summary(summary, step_sum + 1)
                img_trace, summary = sess.run([sampler, g_trace], feed_dict={noise_input: sample_noise})
                summary_writer.add_summary(summary, step_sum + 1)
                
                save_images('output/pred_%d_steps.png' % (step_sum + 1), rescale_to_rgb(img_pred))
                save_images('output/trace_%d_steps.png' % (step_sum + 1), rescale_to_rgb(img_trace))

            if (step_sum + 1) % save_weights_per_step == 0:
                saver = save_model(sess, saver, 'model', step_sum + 1)

            step_sum += 1
            # end step for
        # end epoch for
    saver = save_model(sess, saver, 'model', step_sum)
    prev_step = step_sum


def main():
    image_width = 100
    image_height = 100
    noise_dim = 100
    g_filter = 128
    d_filter = 128
    batch_size = 32
    training_set_path = 'train'
    channel_count = 3
    adam_lr = 0.0002
    adam_beta1 = 0.5
    random_count = 64
    sample_count = 64
    
    test_generator_per_step = 100
    save_weights_per_step = 500

    print('** LOADING TRAINING SET **')
    train_x = load_training_set(training_set_path)
    channel_count = train_x.shape[-1]
    print(train_x.shape, train_x.dtype)

    print('** CONSTRUCTING VARIABLES AND COMPUTE GRAPH **')
    image_input = tf.placeholder(tf.float32, [None, image_height, image_width, channel_count],
                                 name='discriminator/input')
    noise_input = tf.placeholder(tf.float32, [None, noise_dim], name='generator/input')
    summary_dict = dict()

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

    tf.contrib.slim.model_analyzer.analyze_vars(t_vars, print_info=True)

    np.random.seed(0)
    sample_noise = np.random.uniform(-1.0, 1.0, size=[sample_count, noise_dim])
    import time
    t = int(time.time())
    np.random.seed(t)

    summary_dict['d_pred'] = tf.summary.histogram('d_pred', tf.reshape(tf.concat([D, GAN], axis=0), [1, -1]))
    summary_dict['g_pred'] = tf.summary.image('g_pred', G, max_outputs=random_count)
    summary_dict['g_trace'] = tf.summary.image('g_trace', G, max_outputs=sample_count)

    print("** STARTING SESSION **")
    saver = tf.train.Saver()
    sess = set_vram_growth()
    tf.global_variables_initializer().run(session=sess)
    summary_writer = tf.summary.FileWriter('log', sess.graph)

    saver = load_model(sess, saver, 'model')

    # start training process
    print('** TRAINING PROCESS STARTED **')
    print('Press "Q" to exit this program')

    train_model(sess, saver, train_x, noise_dim, random_count, d_opt, g_opt, Sampler, image_input, noise_input, sample_noise,
                summary_dict, summary_writer, batch_size=batch_size, test_generator_per_step=test_generator_per_step, save_weights_per_step=save_weights_per_step)

    sess.close()
    summary_writer.close()


if __name__ == '__main__':
    main()
