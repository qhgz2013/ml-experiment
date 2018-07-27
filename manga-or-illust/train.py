import keras.backend as k
import cv2
import os
from tqdm import tqdm
from keras.layers import *
from keras.optimizers import *
from keras.models import Model, load_model
import math


def crop_images(image_path_in, image_path_out, dst_res=(100, 100)):
    if not os.path.exists(image_path_out):
        os.mkdir(image_path_out)
    files = os.listdir(image_path_in)
    for file in tqdm(files):
        if os.path.isdir(os.path.join(image_path_in, file)):
            continue
        image = cv2.imread(os.path.join(image_path_in, file))
        image = cv2.resize(image, dst_res, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(image_path_out, file), image, (cv2.IMWRITE_JPEG_QUALITY, 100))


def yield_training_set(manga_dir, illust_dir, batch_size=64, loop=True):
    manga_files = os.listdir(manga_dir)
    illust_files = os.listdir(illust_dir)
    tag_manga = [0] * len(manga_files)
    tag_illust = [1] * len(illust_files)
    manga_files = [os.path.join(manga_dir, x) for x in manga_files]
    illust_files = [os.path.join(illust_dir, x) for x in illust_files]
    all_files = manga_files + illust_files
    all_tags = tag_manga + tag_illust
    rnd_idx = np.arange(0, len(all_files))
    np.random.shuffle(rnd_idx)
    all_files = [all_files[i] for i in rnd_idx]
    all_tags = [all_tags[i] for i in rnd_idx]

    cache_images = list()
    cache_tags = list()

    while True:
        for (file, tag) in zip(all_files, all_tags):
            img = cv2.imread(file)
            cache_images.append(img)
            cache_tags.append(tag)

            if len(cache_images) == batch_size:
                images = np.array(cache_images, dtype=np.float32) / 255
                tags = np.array(cache_tags, dtype=np.float32)

                yield images, tags

                cache_tags = list()
                cache_images = list()

        if len(cache_images) > 0:
            images = np.array(cache_images, dtype=np.float32) / 255
            tags = np.array(cache_tags, dtype=np.float32)

            yield images, tags

            cache_tags = list()
            cache_images = list()

        if not loop:
            break


def get_training_set_samples(manga_dir, illust_dir):
    manga_files = os.listdir(manga_dir)
    illust_files = os.listdir(illust_dir)
    return len(manga_files) + len(illust_files)


def set_vram_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    k.set_session(sess)


def conv_model(input_shape):
    model_input = Input(input_shape, name='input')
    model = Conv2D(32, (7, 7), strides=(2, 2), padding='same', name='conv1')(model_input)
    model = BatchNormalization(name='bn1')(model)
    model = Activation('relu', name='relu1')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(model)

    model = Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='conv2')(model)
    model = BatchNormalization(name='bn2')(model)
    model = Activation('relu', name='relu2')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='poo2')(model)

    model = Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='conv3')(model)
    model = BatchNormalization(name='bn3')(model)
    model = Activation('relu', name='relu3')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(model)

    model = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv4')(model)
    model = BatchNormalization(name='bn4')(model)
    model = Activation('relu', name='relu4')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(model)

    model = Flatten(name='flatten')(model)
    model = Dense(128, name='fc1')(model)
    model = BatchNormalization(name='bn6')(model)
    model = Activation('relu', name='relu6')(model)
    model = Dense(1, name='fc2')(model)
    model = Activation('sigmoid', name='output')(model)

    return Model(model_input, model, 'cnn_classifier')


def main():
    image_width = 400
    image_height = 400
    image_channel = 3
    batch_size = 32

    input_dims = (image_height, image_width, image_channel)
    training_set_path = 'D:/ML-TRAINING-SET/manga_or_illust_cropped'
    dev_set_path = 'D:/ML-TRAINING-SET/manga_or_illust_cropped'
    model_path = 'cnn_classifier.h5'
    # print('cropping manga')
    # crop_images('D:/ML-TRAINING-SET/manga_or_illust/manga', 'D:/ML-TRAINING-SET/manga_or_illust_cropped/manga',
    #             (image_width, image_height))
    # print('cropping illust')
    # crop_images('D:/ML-TRAINING-SET/manga_or_illust/illust', 'D:/ML-TRAINING-SET/manga_or_illust_cropped/illust',
    #             (image_width, image_height))
    # print('cropping manga-dev')
    # crop_images('D:/ML-TRAINING-SET/manga_or_illust/manga-dev',
    #             'D:/ML-TRAINING-SET/manga_or_illust_cropped/manga-dev',
    #             (image_width, image_height))
    # print('cropping illust-dev')
    # crop_images('D:/ML-TRAINING-SET/manga_or_illust/illust-dev',
    #             'D:/ML-TRAINING-SET/manga_or_illust_cropped/illust-dev',
    #             (image_width, image_height))

    print('Loading dev set')
    dev_x = None
    dev_y = None
    for _x, _y in yield_training_set(os.path.join(dev_set_path, 'manga-dev'),
                                     os.path.join(dev_set_path, 'illust-dev'), 128):
        dev_x = _x
        dev_y = _y
        break
    print('Loading training set count')
    sample_count = get_training_set_samples(os.path.join(training_set_path, 'manga'),
                                            os.path.join(training_set_path, 'illust'))
    print('Staring session and building models')
    set_vram_growth()
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = conv_model(input_dims)
    model.summary()
    opt = Adam(0.001)
    model.compile(opt, 'binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(yield_training_set(os.path.join(training_set_path, 'manga'),
                                           os.path.join(training_set_path, 'illust'), batch_size),
                        epochs=3, validation_data=(dev_x, dev_y),
                        steps_per_epoch=int(math.ceil(sample_count / batch_size)))
    # opt.lr = 0.0004
    k.set_value(opt.lr, 0.0004)
    model.fit_generator(yield_training_set(os.path.join(training_set_path, 'manga'),
                                           os.path.join(training_set_path, 'illust'), batch_size),
                        epochs=3, validation_data=(dev_x, dev_y),
                        steps_per_epoch=int(math.ceil(sample_count / batch_size)))

    k.set_value(opt.lr, 0.0001)
    model.fit_generator(yield_training_set(os.path.join(training_set_path, 'manga'),
                                           os.path.join(training_set_path, 'illust'), batch_size),
                        epochs=3, validation_data=(dev_x, dev_y),
                        steps_per_epoch=int(math.ceil(sample_count / batch_size)))

    k.set_value(opt.lr, 0.00001)
    model.fit_generator(yield_training_set(os.path.join(training_set_path, 'manga'),
                                           os.path.join(training_set_path, 'illust'), batch_size),
                        epochs=5, validation_data=(dev_x, dev_y),
                        steps_per_epoch=int(math.ceil(sample_count / batch_size)))
    model.save(model_path)


if __name__ == '__main__':
    main()
