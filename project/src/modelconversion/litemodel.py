import numpy as np
import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from project import config
from project.src.execution.modelfile import load

# #model_path
# model_path = 'C:/Users/Aanand/PycharmProjects/OD/Project_1/SRC/MobileNet_model/mobilenet_model.h5'
# model_lite = tf.keras.models.load_model(model_path)
# #model_lite.summary()


def tf_lite():
    model_lite = load(config.MODEL)

    #converting
    converter = tf.lite.TFLiteConverter.from_keras_model(model_lite)
    tflite_model = converter.convert()
    open("tflite_model.tflite", "wb").write(tflite_model)

tf_lite()