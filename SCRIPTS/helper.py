# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:33:16 2019

@author: kennedy
"""

import re
import codecs
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(y_test, y_pred, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(np.array(y_test), np.array(y_pred))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("accuracy: {}".format(accuracy_score(y_test, y_pred)))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class StsaParser(object):
    """
    parser to get the Stsa dataset
    """
    def __init__(self):
        pass
    def transform_label_to_numeric(self, y):
            if '1' in y:
                return 1
            else:
                return 0

    def parse_line(self, row):
        row = row.split(' ')
        text = (' '.join(row[1:]))
        label = self.transform_label_to_numeric(row[0])
        return (re.sub(r'\W+', ' ', text), label)

    def get_data(self, file_path):
        data = []
        labels = []
        f = codecs.open(file_path, 'r', encoding = "utf8",errors = 'ignore')
        for line in f:
            doc, label = self.parse_line(line)
            data.append(doc)
            labels.append(label)
        return data, np.array(labels)

        

        