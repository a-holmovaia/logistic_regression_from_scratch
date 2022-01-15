import pandas as pd
import numpy as np
import math


class LogisticRegressionCustom:
    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return 1 / (1 + math.pow(math.e, -np.sum(t)))

    def predict_proba(self, X):
        t = np.multiply(self.coef_, X)
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        if self.fit_intercept:
            self.coef_ = [0] * (X_train.shape[1] + 1)
        else:
            self.coef_ = [0] * (X_train.shape[1])
        mse_error = [[], []]
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                if self.fit_intercept:
                    row = np.insert(row, 0, 1.0)
                y_hat = self.predict_proba(row)
                sub = -self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                for j in range(len(self.coef_)):
                    self.coef_[j] = self.coef_[j] + sub * row[j]
                if _ == 0:
                    mse_error[0].append(1 / len(X_train) * pow((y_hat - y_train[i]), 2))
                if _ == self.n_epoch - 1:
                    mse_error[1].append(1 / len(X_train) * pow((y_hat - y_train[i]), 2))
        return mse_error

    def fit_log_loss(self, X_train, y_train):
        if self.fit_intercept:
            self.coef_ = [0] * (X_train.shape[1] + 1)
        else:
            self.coef_ = [0] * (X_train.shape[1])
        log_loss_error = [[], []]
        for _ in range(self.n_epoch):
            for i, row in enumerate(X_train):
                if self.fit_intercept:
                    row = np.insert(row, 0, 1.0)
                y_hat = self.predict_proba(row)
                for j in range(len(self.coef_)):
                    self.coef_[j] = self.coef_[j] - (self.l_rate * (y_hat - y_train[i]) * row[j]) / X_train.shape[0]
                if _ == 0:
                    log_loss_error[0].append(
                        -1 / len(X_train) * (y_train[i] * math.log(y_hat) + (1 - y_train[i]) * math.log(1 - y_hat)))
                if _ == self.n_epoch - 1:
                    log_loss_error[1].append(
                        -1 / len(X_train) * (y_train[i] * math.log(y_hat) + (1 - y_train[i]) * math.log(1 - y_hat)))
        return log_loss_error

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            if self.fit_intercept:
                row = np.insert(row, 0, 1.0)
            y_hat = self.predict_proba(row)
            if y_hat <= cut_off:
                predictions.append(0)
            else:
                predictions.append(1)
        return predictions
