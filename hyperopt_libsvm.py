from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from hyperopt import hp, tpe, STATUS_OK, fmin
from sklearn.metrics import accuracy_score, f1_score, classification_report
from file_utils import *

import numpy as np
import time
import os


class HyperoptTuner(object):

    def __init__(self, classifier, cargs, train_X=None, train_y=None, test_X=None, test_y=None, cluster_id=None, base_dir=None):
        self.classifier = classifier
        self.cargs = cargs
        self.poly_namespace = ''
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.cluster_id = cluster_id
        self.best_acc = .0
        self.best_f1 = .0
        self.best_iter = 0
        self.cnt = 0
        self.best_cfg = None
        self.clf_report = ""
        self.pred_results = []
        self.elapsed_time = None
        self.base_dir = base_dir
        self.correct = 0
        self.svc_package = self.select_svc_package(classifier)

    def select_svc_package(self, classifier):
        if classifier == 'SVC':
            if self.cargs['thundersvm']:
                from thundersvm import SVC
                self.poly_namespace = 'polynomial'
            else:
                from sklearn.svm import SVC
                self.poly_namespace = 'poly'
            return SVC
        return None

    # pre-set parameters space
    def _preset_ps(self):
        if self.classifier == 'SVC':
            params = {
                'C': hp.uniform('C', 2 ** 10, 2 ** 20),
                # NOTE: CHANGE ALL KERNEL FROM 'POLY' TO 'POLYNOMIAL'
                # , 'linear', 'rbf', 'polynomial'
                'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', self.poly_namespace]),
                'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
                # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
                'degree': hp.choice('degree', [i for i in range(1, 6)]),
                'coef0': hp.uniform('coef0', 1, 10),
                'class_weight': hp.choice('class_weight', ['balanced', None]),
            }
            params = self._svm_constraint(params)
        if self.classifier == 'RF':
            params = {
                'n_estimators': hp.choice('n_estimators', np.arange(100, 500, dtype=int)),
                'max_depth': hp.choice('max_depth', np.arange(5, 20, dtype=int)),
                'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, dtype=int)),
                'min_samples_split': hp.choice('min_samples_split', np.arange(2, 6, dtype=int))
            }
        if self.classifier == 'KNN':
            params = {
                'n_neighbors': hp.choice('n_neighbors', [3, 5, 7]),
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'leaf_size': hp.choice('leaf_size', np.arange(30, 50, dtype=int)),
                'p': hp.choice('p', [1,2])
            }
        if self.classifier == 'MLP':
            params = {}
        if self.classifier == 'GP':
            params = {
                'kernel': 1.0 * RBF(1.0)
            }
        if self.classifier == 'DT':
            params = {
                'max_depth': hp.choice('max_depth', np.arange(5, 20, dtype=int)),
                'min_samples_leaf': hp.choice('min_samples_leaf', np.arange(1, 5, dtype=int)),
                'min_samples_split': hp.choice('min_samples_split', np.arange(2, 6, dtype=int)),
                'class_weight': hp.choice('class_weight', ['balanced', None]),
            }
        if self.classifier == 'ADAB':
            params = {
                'n_estimators': hp.choice('n_estimators', np.arange(50, 200, dtype=int)),
                'learning_rate': hp.uniform('learning_rate',  0.01, 10)
            }
        if self.classifier == 'GNB':
            params = {}
        if self.classifier == 'QDA':
            params = {}
        return params

    def _svm_constraint(self, params):
        if params['kernel'] != self.poly_namespace:
            params.pop('degree', None)

        if params['kernel'] != self.poly_namespace and params['kernel'] != 'sigmoid':
            params.pop('coef0', None)

        if params['kernel'] == 'linear':
            params.pop('gamma', None)

        return params

    def classifier_from_string(self, params):
        if self.classifier == 'SVC':
            clf = self.svc_package(**params, random_state=42)
        if self.classifier == 'RF':
            clf = RandomForestClassifier(**params, random_state=42)
        if self.classifier == 'KNN':
            clf = KNeighborsClassifier(**params)
        if self.classifier == 'MLP':
            clf = MLPClassifier(**params, random_state=42)
        if self.classifier == 'GP':
            clf = GaussianProcessClassifier(**params, random_state=42)
        if self.classifier == 'DT':
            clf = DecisionTreeClassifier(**params, random_state=42)
        if self.classifier == 'ADAB':
            clf = AdaBoostClassifier(**params, random_state=42)
        if self.classifier == 'GNB':
            clf = GaussianNB(**params, random_state=42)
        if self.classifier == 'QDA':
            clf = QuadraticDiscriminantAnalysis(**params, random_state=42)
        return clf

    def _clf(self, params, is_tuning=True):
        clf = self.classifier_from_string(params)
        clf.fit(self.train_X, self.train_y)
        pred = clf.predict(self.test_X)
        self.pred_results = pred
        score_acc = accuracy_score(self.test_y, pred)
        score_f1 = f1_score(self.test_y, pred, average='macro')

        self.cnt += 1
        if score_acc > self.best_acc:
            self.best_acc = score_acc
            self.best_f1 = score_f1
            self.best_cfg = params
            self.best_iter = self.cnt
            self.clf_report = str(classification_report(self.test_y, pred))

        if is_tuning:
            print('current_iter / best_iter: %d / %d' %
                  (self.cnt, self.best_iter))
        else:
            correct = 0
            for pred_y, true_y in zip(pred, self.test_y):
                if pred_y == true_y:
                    correct += 1
            self.correct = correct
        return score_acc

    def _object2minimize(self, params):
        score_acc = self._clf(params)
        return {'loss': 1 - score_acc, 'status': STATUS_OK}

    def tune_params(self, n_iter=200):
        t_start = time.time()
        fmin(fn=self._object2minimize,
             algo=tpe.suggest,
             space=self._preset_ps(),
             max_evals=n_iter)
        t_end = time.time()
        self.elapsed_time = t_end - t_start
        # print the final optimized result
        self._clf(self.best_cfg, is_tuning=False)

    def optimized_svm(self, params):
        self._clf(params, False)
