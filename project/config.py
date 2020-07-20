import os

PROJECT_HOME = '/home/eml/Object_classification/project'
#print(os.path.exists(PROJECT_HOME))

DATA = os.path.join(PROJECT_HOME,'data')
#print(DATA)
#print(os.path.exists(DATA))

PROCESSED_DATASET = os.path.join(DATA,'dataset')
#print(PROCESSED_DATASET)
#print(os.path.exists(PROCESSED_DATASET))

TRAIN_SET = os.path.join(PROCESSED_DATASET,'train')
#print(TRAIN_SET)
#print(os.path.exists(TRAIN_SET))

VALID_SET = os.path.join(PROCESSED_DATASET,'valid')
#print(VALID_SET)
#print(os.path.exists(VALID_SET))

TEST_SET = os.path.join(PROCESSED_DATASET,'test')
#print(TEST_SET)
#print(os.listdir(TEST_SET))
#print(os.path.exists(TEST_SET))

MODEL_FILE = os.path.join(PROJECT_HOME, 'model')
#print(os.path.exists(MODEL_FILE))

MODEL = os.path.join(MODEL_FILE,'Saved_model.h5')
#print(os.path.exists(MODEL))

ONNXMODEL = os.path.join(MODEL_FILE,'onnxmodel.onnx')
#print(os.path.exists(ONNXMODEL))

IMAGE_TESTPATH = os.path.join(DATA, 'testpred')
IMAGE_PRED = os.path.join(IMAGE_TESTPATH, '1.jpg')

SRC = os.path.join(PROJECT_HOME, 'src')
PICKLE = os.path.join(SRC, 'lb.pickle')