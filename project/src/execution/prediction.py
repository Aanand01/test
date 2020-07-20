import matplotlib.pyplot as plt
import warnings
import time
import itertools
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from project.src.imagepreparation.imageprep import img_gen
from project.src.imagepreparation.parameter2 import *
from project.src.execution.modelfile import load
from project import config


def choice():

    user_input = int(input('Enter model choice :'))

    if user_input == 1:
        model_pred = load(config.MODEL)
        return model_pred
    elif user_input == 2:
        model_pred = load(config.ONNXMODEL)
        return model_pred
    else:
        print("Enter user input number either 1 or 2")


def loaded():

    model_prediction = choice()
    #print(model_pred.summary())

    # test_labels
    train_batches, valid_batches, test_batches = img_gen()
    train_st, valid_st, test_st = ss()

    test_label = test_batches.classes
    #print(test_label)

    # test batch class indices(0 : 'cat', 1 : 'dog')
    #print(test_batches.class_indices)

    # prediction the test batch
    predictions_test = model_prediction.predict(test_batches, steps=test_st, verbose=0)
    #print(predictions_test.argmax(axis=1))

    return test_label, predictions_test


def get_metrics():

    test_lab, pred_test = loaded()

    yt = test_lab
    yp = pred_test.argmax(axis=1)

    conmat = confusion_matrix(yt, yp)
    #evaluation metrics
    print(conmat)
    print('--------------------------')
    print ('Accuracy Score :', accuracy_score(yt, yp))
    print('--------------------------')
    print('precision_score :', precision_score(yt, yp, average='weighted'))
    print('--------------------------')
    print('recall :', recall_score(yt, yp, average='weighted'))
    print('--------------------------')
    print('fi score :', f1_score(yt, yp, average='weighted'))
    print('--------------------------')
    print('AUC :', roc_auc_score(yt, yp))
    print('--------------------------')
    print ('Report : ', classification_report(yt, yp))


#infernce time
if __name__ == '__main__':
    start_time = time.time()

    get_metrics()

    print('Took', round(time.time() - start_time, 2), 'seconds')

def roc():

    test_lab, pred_test = loaded()

    yt = test_lab
    yp = pred_test.argmax(axis=1)

    fpr,tpr,thresh = roc_curve(yt,yp)
    plt.plot(fpr, tpr, 'r--')
    plt.plot(fpr, fpr, 'b--')
    plt.show()

#roc()


#defining function to print confusion matrix
def cm(title='Confusion matrix',
                            cmap=plt.cm.Blues):

    test_lab, pred_test = loaded()

    yt = test_lab
    yp = pred_test.argmax(axis=1)

    con_mat = confusion_matrix(yt, yp)

    #label ploting
    cm_classes = ['cat', 'dog']

    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm_classes))
    plt.xticks(tick_marks, cm_classes, rotation=45)
    plt.yticks(tick_marks, cm_classes)

    thresh = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, con_mat[i, j],
            horizontalalignment="center",
            color="white" if con_mat[i, j] > thresh else "black")

    #confusion matrix
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#cm()