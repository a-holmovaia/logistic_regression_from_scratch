from LogisticRegressionCustom import LogisticRegressionCustom
import pandas as pd
import sklearn.model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    data = load_breast_cancer()
    columns = ['worst concave points', 'worst perimeter', 'worst radius']
    df = pd.DataFrame(data.data, columns=data.feature_names).loc[:, columns]
    target = data.target
    scaler = StandardScaler().fit(df)
    df = scaler.transform(df)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, target, train_size=0.8,
                                                                                random_state=43)

    mse_regressor = LogisticRegressionCustom(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    mse_error = mse_regressor.fit_mse(X_train, y_train)
    y_pred_mse = mse_regressor.predict(X_test)
    acc_mse = accuracy_score(y_test, y_pred_mse)

    log_loss_regressor = LogisticRegressionCustom(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    log_loss_error = log_loss_regressor.fit_log_loss(X_train, y_train)
    y_pred_logloss = log_loss_regressor.predict(X_test)
    acc_logloss = accuracy_score(y_test, y_pred_logloss)

    sklearn_regressor = LogisticRegression(fit_intercept=True, max_iter=1000).fit(X_train, y_train)
    y_pred_sklearn = sklearn_regressor.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)

    keys = ['mse_accuracy', 'logloss_accuracy', 'sklearn_accuracy', 'mse_error_first', 'mse_error_last',
            'logloss_error_first', 'logloss_error_last']
    values = [acc_mse, acc_logloss, acc_sklearn, mse_error[0], mse_error[1], log_loss_error[0], log_loss_error[1]]
    dic = dict(zip(keys, values))
    print(dic)
