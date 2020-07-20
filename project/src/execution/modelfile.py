import tensorflow as tf
from tensorflow.keras.models import load_model

def mc():
    mobile = tf.keras.applications.mobilenet.MobileNet()
    return mobile


def save(model):
    model.save('Saved_model.h5')


def load(file):
    model_pred = load_model(file)
    return model_pred
