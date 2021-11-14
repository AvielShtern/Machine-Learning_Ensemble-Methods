########################################################################################
# FILE: adaboost.py
# WRITER : Aviel Shtern
# LOGIN : aviel.shtern
# ID: 206260499
# EXERCISE : Introduction to Machine Learning: Exercise 4 - PAC & Ensemble Method 2021
########################################################################################

import numpy as np
import math


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D = np.full(X.shape[0], 1 / X.shape[0])
        for i in range(self.T):
            cur_model = self.WL(D, X, y)
            cur_predict = cur_model.predict(X)
            cur_epsilon = np.sum(D * (y != cur_predict))
            cur_weight = 0.5 * math.log((1 / cur_epsilon) - 1)
            D = D * np.exp(-1 * y * cur_weight * cur_predict)
            D = D / np.sum(D)
            self.h[i] = cur_model
            self.w[i] = cur_weight
        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        predict = np.zeros(X.shape[0])
        for i in range(max_t):
            predict = predict + self.w[i] * self.h[i].predict(X)
        return np.sign(predict)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        return np.sum(y != self.predict(X, max_t)) / X.shape[0]
