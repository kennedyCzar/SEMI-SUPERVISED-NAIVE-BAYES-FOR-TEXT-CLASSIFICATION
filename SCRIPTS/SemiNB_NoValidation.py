# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 04:51:16 2019

@author: kennedy
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import codecs
import os
import numpy as np
from mimetypes import guess_type
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class docparser(object):
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

    def get_data(self, file_path, Text = True):
        if Text:
          data = []
          labels = []
          f = codecs.open(file_path, 'r', encoding = "utf8",errors = 'ignore')
          for line in f:
              doc, label = self.parse_line(line)
              data.append(doc)
              labels.append(label)
          return data, np.array(labels)
        else:
          import pandas as pd
          df = pd.read_csv(file_path)
          df.columns = ['Label', 'Rating', 'Review']
          rating = pd.get_dummies(df.loc[:, 'Rating'])
          text = []
          labels = []
          for ii, ij in zip(df.loc[:, 'Review'].values, df.loc[:, 'Label'].values):
            #print(ii, ij)
            if ii == ' ':
              pass
            else:
              text.append(ii)
              labels.append(ij)
          return text, np.array(labels), rating
  

    def shuffle_dataset(X, y, seed=None):
            """ Random shuffle of the samples 
            in X and y 
            """
            if seed:
                np.random.seed(seed)
            idx = [ii for ii in range(X.shape[0])]
            np.random.shuffle(idx)
            return X[idx], y[idx]
      
    def split(data, labels, test_size = 0.3,shuff = True, seed = None):
        '''
        :params
          --label: labels of the dataset
          --rating: rating as dummies categorical variables
          --text: text data
        :Returntype:
           X_supervised:
           X_unsupervised:
           y_supervised:
        '''
        X = np.array(data)
        y = np.array(labels)
        if shuff:
          from sklearn.utils import shuffle
          X, y = shuffle_dataset(X, y, seed)
        else:
          split = len(y) - int(len(y) // (1/test_size))
          X_supvsd, X_unsupvsd = X[:split], X[split:]
          y_supvsd, y_supvsd = y[:split], y[split:]
          return X_supvsd, X_unsupvsd, y_supvsd, y_supvsd
      
 #%% Semisupervised NB Classifier
 
class NaiveBayesSemiSupervised(object):
    """
    This class implements a modification of the Naive Bayes classifier
    in order to deal with unlabelled data. We use an Expectation-maximization 
    algorithm (EM). 
    This work is based on the paper
    'Semi-Supervised Text Classification Using EM' by
    Kamal Nigam Andrew McCallum Tom Mitchell
    available here:
    https://www.cs.cmu.edu/~tom/pubs/NigamEtAl-bookChapter.pdf
    """
    def __init__(self, max_features=None, max_rounds=50, tolerance=1e-6):
        """
        constructor for NaiveBayesSemiSupervised object
        keyword arguments:
            -- max_features: maximum number of features for documents vectorization
            -- max_rounds: maximum number of iterations for EM algorithm
            -- tolerance: threshold (in percentage) for total log-likelihood improvement during EM
        """
        self.max_features = max_features
        self.n_labels = 0
        self.max_rounds = max_rounds
        self.tolerance = tolerance
          
          
    def train(self, X_supervised, X_unsupervised, y_supervised, y_unsupervised):
        """
        train the modified Naive bayes classifier using both labelled and 
        unlabelled data. We use the CountVectorizer vectorizaton method from scikit-learn
        positional arguments:
            -- X_supervised: list of documents (string objects). these documents have labels
                example: ["all parrots are interesting", "some parrots are green", "some parrots can talk"]
            -- X_unsupervised: list of documents (string objects) as X_supervised, but without labels
            -- y_supervised: labels of the X_supervised documents. list or numpy array of integers. 
                example: [2, 0, 1, 0, 1, ..., 0, 2]
            -- X_supervised, X_unsupervised, y_supervised, y_unsupervised
        """
        count_vec = CountVectorizer(max_features = self.max_features)
        count_vec.fit(X_supervised)
        self.n_labels = len(set(y_supervised))
        if self.max_features is None:
            self.max_features = len(count_vec.vocabulary_)
        X_supervised = np.asarray(count_vec.transform(X_supervised).todense())
        X_unsupervised = np.asarray(count_vec.transform(X_unsupervised).todense())
        #train Naive Bayes
        self.train_naive_bayes(X_supervised, y_supervised)
        predi = self.predict(X_supervised)
        old_likelihood = 1
        cumulative_percent = 0
        while self.max_rounds > 0:
            self.max_rounds -= 1
            predi = self.predict(X_unsupervised)
            self.train_naive_bayes(X_unsupervised, predi)
            predi = self.predict(X_unsupervised)
            correct = 0
            for ij in predi:
              if ij == 1:
                correct += 1
            correct_percent = correct/len(X_unsupervised)
            cumulative_percent += correct_percent
            print(str(correct_percent) + "%")
            total_likelihood = self.get_log_likelihood( X_supervised, X_unsupervised, y_supervised)
            print("total likelihood: {}".format(total_likelihood))
            if self._stopping_time(old_likelihood, total_likelihood):
                print('log-likelihood not improved..Stopping EM at round %s'%self.max_rounds)
                break
            old_likelihood = total_likelihood.copy()
            
    def _stopping_time(self, old_likelihood, new_likelihood):
        """
        returns True if there is no significant improvement in log-likelihood and false else
        positional arguments:
            -- old_likelihood: log-likelihood for previous iteration
            -- new_likelihood: new log-likelihood
        """
        relative_change = np.absolute((new_likelihood-old_likelihood)/old_likelihood) 
        if (relative_change < self.tolerance):
            print("stopping time")
            return True
        else:
            return False
          
    def get_log_likelihood(self, X_supervised, X_unsupervised, y_supervised):
        """
        returns the total log-likelihood of the model, taking into account unsupervised data
        positional arguments:
            -- X_supervised: list of documents (string objects). these documents have labels
                example: ["all parrots are interesting", "some parrots are green", "some parrots can talk"]
            -- X_unsupervised: list of documents (string objects) as X_supervised, but without labels
            -- y_supervised: labels of the X_supervised documents. list or numpy array of integers. 
                example: [2, 0, 1, 0, 1, ..., 0, 2]
        """
        unsupervised_term = np.sum(self._predict_proba_unormalized(X_unsupervised), axis=1)
        unsupervised_term = np.sum(np.log(unsupervised_term))
        supervised_term = self._predict_proba_unormalized(X_supervised)
        supervised_term = np.take(supervised_term, y_supervised)
        supervised_term = np.sum(np.log(supervised_term))
        total_likelihood = supervised_term + unsupervised_term
        return total_likelihood

    def word_proba(self, X, y, c):
        """
        returns a numpy array of size max_features containing the conditional probability
        of each word given the label c and the model parameters
        positional arguments:
            -- X: data matrix, 2-dimensional numpy ndarray
            -- y: numpy array of labels, example: np.array([2, 0, 1, 0, 1, ..., 0, 2])
            -- c: integer, the class upon which we condition
        """
        numerator = 1 + np.sum( X[np.equal( y, c )], axis=0)
        denominator = self.max_features + np.sum( X[ np.equal( y, c)])
        return np.squeeze(numerator)/denominator

    def class_proba(self, X, y, c):
        """
        returns a numpy array of size n_labels containing the conditional probability
        of each label given the label model parameters
        positional arguments:
            -- X: data matrix, 2-dimensional numpy ndarray
            -- y: numpy array of labels, example: np.array([2, 0, 1, 0, 1, ..., 0, 2])
            -- c: integer, the class upon which we condition
        """
        numerator = 1 + np.sum( np.equal( y, c) , axis=0)
        denominator = X.shape[0] + self.n_labels
        return numerator/denominator

    def train_naive_bayes(self, X, y):
        """
        train a regular Naive Bayes classifier
        positional arguments:
             -- X: data matrix, 2-dimensional numpy ndarray
             -- y: numpy array of labels, example: np.array([2, 0, 1, 0, 1, ..., 0, 2])
        """
        word_proba_array = np.zeros((self.max_features, self.n_labels))
        for c in range(self.n_labels):
            word_proba_array[:,c] = self.word_proba( X, y, c)
        labels_proba_array = np.zeros(self.n_labels)
        for c in range(self.n_labels):
            labels_proba_array[c] = self.class_proba( X, y, c)
        self.word_proba_array = word_proba_array
        self.labels_proba_array = labels_proba_array

    def _predict_proba_unormalized(self, X_test):
        """
        returns unormalized predicted probabilities (useful for log-likelihood computation)
        positional arguments:
             -- X: data matrix, 2-dimensional numpy ndarray
        """
        proba_array_unormalized = np.zeros((X_test.shape[0], self.n_labels))
        for c in range(self.n_labels):
            temp = np.power(np.tile(self.word_proba_array[:,c], (X_test.shape[0] ,1)), X_test)
            proba_array_unormalized[:,c] = self.labels_proba_array[c] * np.prod(temp, axis=1)
        return proba_array_unormalized

    def predict_proba(self, X):
        """
        returns model predictions (probability)
        positional arguments:
             -- X: data matrix, 2-dimensional numpy ndarray
        """
        proba_array_unormalized = self._predict_proba_unormalized(X)
        proba_array = np.true_divide(proba_array_unormalized, np.sum(proba_array_unormalized, axis=1)[:, np.newaxis])
        return proba_array

    def predict(self, X):
        """
        returns model predictions (class labels)
        positional arguments:
             -- X: data matrix, 2-dimensional numpy ndarray
        """
        return np.argmax(self.predict_proba( X), axis=1)
      
if __name__ == '__main__':
    import os
    import random
    random.seed(23)
    from sklearn.model_selection import KFold, train_test_split
    Text = False
    max_features=None
    NSPLIT = 4
    n_sets = 4
    set_size = 1.0 / n_sets
    cumulative_percent = 0
    #set project directory
    os.chdir('D:\\FREELANCER\\SEMI_NB_TEXT_CLASSIFICATION')
    file_path = os.path.join('DATASET','stas_train.text')
    if Text:
      data, labels = docparser().get_data(os.path.join('DATASET','stas-train.txt'), Text)
    else:
      data, labels, rating = docparser().get_data(os.path.join('DATASET','mytracks_NaiveBayes_Filter.csv'), Text)
    X_supervised, X_unsupervised, y_supervised, y_unsupervised = train_test_split(data, labels, test_size = 0.8)
    #Max features should be left as it is 5738
    clf = NaiveBayesSemiSupervised(max_features)
    #train and evaluate accuracy
    clf.train(X_supervised, X_unsupervised, y_supervised, y_unsupervised)




      
      