import onnx
import keras2onnx

from tensorflow.keras.models import load_model
from project import config


def ox(mod):
    #onnx model name
    onnx_model_name = 'onnxmodel.onnx'
    model = load_model(mod)
    #print(model.summary())
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx.save_model(onnx_model, onnx_model_name)
    print('onnx model is created and saved')

ox(config.MODEL)
