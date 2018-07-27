from keras.models import load_model
import keras.backend as k
import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request
import base64


app = Flask(__name__)
model = None


def set_vram_growth():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    k.set_session(sess)


@app.route('/classify', methods=['POST'])
def process_request():
    img_binary = request.form['image']
    image_width = 400
    image_height = 400
    if img_binary is None:
        return '-1'
    img_binary = base64.decodebytes(bytes(img_binary, 'utf-8'))
    nparr = np.fromstring(img_binary, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
    x = np.array([img], dtype=np.float32) / 255
    y = model.predict(x)
    return str(y[0, 0])


def main():
    model_path = 'cnn_classifier.h5'

    set_vram_growth()
    global model
    model = load_model(model_path)

    app.run('::1', 10087)


if __name__ == '__main__':
    main()
