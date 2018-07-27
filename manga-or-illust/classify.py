from keras.models import load_model
import keras.backend as k
import tensorflow as tf
import cv2
import numpy as np
import shutil
import os
from tqdm import tqdm


def set_vram_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    k.set_session(sess)


def list_dir(path):
    ret_list = list()
    data = os.listdir(path)
    for file in data:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            ret_list.append(file_path)
        else:
            ret_list += list_dir(file_path)
    return ret_list


def main():
    files = list_dir('D:/PixivRanking')
    model = load_model('cnn_classifier.h5')
    temp_image_cache = list()
    start_idx = 0
    for file, i in zip(tqdm(files), range(len(files))):
        try:
            img = cv2.imread(file)
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
        except Exception as ex:
            print('error while reading %s' % file)
            print(ex.args)
            img = np.zeros((400, 400, 3))
        temp_image_cache.append(img)

        if len(temp_image_cache) == 128:
            x = np.array(temp_image_cache, np.float32) / 255
            y = model.predict(x)
            temp_image_cache = list()

            for t in range(128):
                if y[t, 0] >= 0.5:
                    src = files[start_idx + t]
                    filename = src.replace('\\', '/').split('/')[-1]
                    dst = os.path.join('D:/ML-TRAINING-SET/illust', filename)
                    shutil.copyfile(src, dst)
            start_idx = i

    if len(temp_image_cache) > 0:
        x = np.array(temp_image_cache, np.float32) / 255
        y = model.predict(x)
        for t in range(len(temp_image_cache)):
            if y[t, 0] >= 0.9:
                src = files[start_idx + t]
                filename = src.replace('\\', '/').split('/')[-1]
                dst = os.path.join('D:/ML-TRAINING-SET/illust', filename)
                shutil.copyfile(src, dst)


def delete_error_image():
    files = list_dir('D:/PixivRanking')
    for file in tqdm(files):
        img = cv2.imread(file)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            import os
            os.remove(file)
            print("removed %s" % file)


if __name__ == '__main__':
    delete_error_image()
