from keras.models import load_model
import keras.backend as k
import tensorflow as tf
import cv2
import numpy as np
import math


def set_vram_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    k.set_session(sess)


def main():
    image_in_list = list()
    while True:
        image_in = input("Input image path (empty to exit): ")

        if image_in is None or image_in == '':
            break
        image_in_list.append(image_in)
    image_width = 400
    image_height = 400
    model_path = 'cnn_classifier.h5'

    set_vram_growth()
    model = load_model(model_path)
    img_list = list()
    for image_in in image_in_list:
        img = cv2.imread(image_in)
        img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
        img_list.append(img)

    x = np.array(img_list, dtype=np.float32) / 255
    y = model.predict(x)

    for path, i in zip(image_in_list, range(len(image_in_list))):
        print(path, ": manga" if y[i, 0] < 0.5 else "illust", '(%f)' % (0.5 + math.fabs(y[i, 0] - 0.5)))


if __name__ == '__main__':
    main()
