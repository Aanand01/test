import numpy as np
import pickle
import cv2
import imutils
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from project import config

# #required
# 1.model path
# 2.labelbiz
# 3.input image

def imgld():
    # load the image
    #path = '/home/eml/Object_classification/project/data/testpred/1.jpg'
    image = cv2.imread(config.IMAGE_PRED)
    output = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0 # normalizing the input image
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    #print(image)
    return output, image

def label():
    labels = ['cat', 'dog']
    labels = np.array(labels)
    lb = LabelBinarizer()
    #print(labels)
    return lb

def lbbin():
    # label binarizer
    lb = label()
    out = open('lb.pickle', "wb")
    out.write(pickle.dumps(lb))
    out.close()

def modelload():
    model = load_model(config.MODEL)
    lb = pickle.loads(open(config.PICKLE, "rb").read())
    return model, lb

def pred():
    output, image = imgld()
    model, lb = modelload()

    # classify the input image
    pred_proba = model.predict(image)[0]
    idx = np.argmax(pred_proba)
    labels = lb.classes_[idx]
    # print(idx) #(0 or 1)
    # print(label) #(cat or dog)

    # displaying image with label and probability score
    label = '{}: {:.2f}%'.format(labels, pred_proba[idx] * 100)
    output = imutils.resize(output, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # show the output image
    print("{}".format(label))
    cv2.imshow("Output", output)
    plt.show()
    cv2.waitKey(0)

pred()


