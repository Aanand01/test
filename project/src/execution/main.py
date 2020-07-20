import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from project.src.execution.modelfile import mc, save
from project.src.imagepreparation.imageprep import img_gen
from project.src.imagepreparation.parameter2 import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




# #'mobile' variable obtains the copy of a single pretrained mobilenet model and weights were saved from being..
# # ..trained on imagenet images
# mobile = keras.applications.mobilenet.MobileNet()
# #mobile.summary()

def model():

    mobile_mod = mc()
    #print(mobile_mod.summary())
    #print(len(mobile_mod.layers))
    x = mobile_mod.layers[-6].output # fifth last layer will be poped to x
    #print(x)
    return mobile_mod, x


def prediction():

    mod, X = model()
    ly_pred = Dense(2, activation='softmax')(X)
    #print(ly_pred)
    #input layer
    in_put = mod.input
    #print(in_put)
    return in_put, ly_pred,

def model_tune():
    #model fitting

    epochs = 20

    earlystop = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks = [earlystop]

    train_batches, valid_batches, test_batches = img_gen()
    train_st, valid_st, test_st = ss()

    INP, OUP = prediction()
    main_model = Model(inputs=INP, outputs=OUP)  # from input layer to sixth last layer
    # new model summay and length
    #main_model.summary()
    #print(len(main_model.layers))

    # we are only changing(updating) the weights of last 5 layers
    for layer in main_model.layers[:-5]:
        layer.trainable = False

    # #compiling the model
    main_model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    main_model.fit_generator(train_batches, steps_per_epoch=train_st,
                            validation_data=valid_batches, validation_steps=valid_st,
                            epochs=epochs,
                            verbose=2, callbacks=callbacks)

    #saving the built model
    save(main_model)

model_tune()