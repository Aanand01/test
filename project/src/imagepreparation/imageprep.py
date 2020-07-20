import tensorflow as tf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from project import config
from project.src.imagepreparation.parameter1 import img_size, batch_size


#current path
#print(os.getcwd())



def tc():
    IMG_SIZE =  img_size(244,244)

    train = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.TRAIN_SET,
        target_size = IMG_SIZE)

    valid = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.VALID_SET,
        target_size = IMG_SIZE)

    test = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.TEST_SET,
        target_size = IMG_SIZE)

    tr = len(train.classes) # getting count of total images in train dataset
    #print(tr)
    vd = len(valid.classes) # getting count of total images in valid dataset
    tt = len(test.classes) # getting count of total images in test dataset

    return tr, vd, tt

#Image generator
#pre-processing dunction is "mobilenet.preprocess_input"

def img_gen():


    IMG_SIZE = img_size(244, 244)
    train_b, valid_b, test_b = batch_size(20, 20, 10)

    train_gen = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.TRAIN_SET,
        target_size = IMG_SIZE,
        batch_size = train_b)

    valid_gen = ImageDataGenerator(
        preprocessing_function = tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.VALID_SET,
        target_size = IMG_SIZE,
        batch_size = valid_b)

    test_gen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory = config.TEST_SET,
        target_size = IMG_SIZE,
        batch_size = test_b,
        shuffle = False)

    return train_gen, valid_gen, test_gen