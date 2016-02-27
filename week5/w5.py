import pandas
import numpy as np

import sklearn.cross_validation as cv
from sklearn.ensemble import \
    RandomForestRegressor, \
    GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

__author__ = 'oderor'


# File header fields:
SEX = 'Sex'
RINGS = 'Rings'


def readdata(fin):
    return pandas.read_csv(fin, header=None)


def readdatawithheader(fin):
    return pandas.read_csv(fin)


def writefile(str, filename):
    fout = open(filename, 'w')
    print(filename, ':', str)
    print(str, file=fout, end="")
    fout.close()


def rf_for_abalone_age(data, RF_EPSILON=0.52):
    # Preprocess 'Sex' column: 'M' -> 1, 'F' -> -1, 'I' -> 0
    data[SEX] = data[SEX].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

    X = data[data.columns.values[:-1]].values
    y = data[RINGS].values.ravel()
    N = len(y)

    # Create a KFold cross-validator
    cvkfold = cv.KFold(N, n_folds=5, shuffle=True, random_state=1)
    # Try multiple numbers of trees in RF. Need to find the minimum t when r2-metrics becomes > 0.52
    print('# of trees\tr2-score')
    for t in range(1, 51):
        # Create RF classifier and fit it using input data
        clf = RandomForestRegressor(n_estimators=t, random_state=1)
        cvscores = cv.cross_val_score(clf, X, y, scoring='r2', cv=cvkfold)
        avgcvscore = np.mean(cvscores)
        print('%10d\t%7.4f' % (t, avgcvscore))
        if avgcvscore > RF_EPSILON:
            break
    return t


def gb(data):
    X = data[data.columns.values[1:]].values
    y = data[data.columns.values[:1]].values.ravel()
    N = len(y)

    X_train, X_test, y_train, y_test = \
        cv.train_test_split(X, y,
                            test_size=0.8,
                            random_state=241)

    # ------------------------------------------------------
    # Deal with Gradient Boosting
    # ------------------------------------------------------

    # Reserve an array to store iteration with min log_loss for each learning rate
    min_iterations_train = []
    min_iterations_test = []


    # Fit Gradient Boosting Classifiers with different learning rates
    learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
    for lr in learning_rates:
        print("GB learning rate = ", lr)

        # Fit the classifier
        gbclf = GradientBoostingClassifier(n_estimators=250,
                                           verbose=True,
                                           random_state=241,
                                           learning_rate=lr)
        gbclf.fit(X_train, y_train)

        # Get log_loss errors after every iteration of the Gradient Boosting
        y_train_pred = gbclf.staged_decision_function(X_train)
        log_loss_train = []
        for y_t_p in y_train_pred:
            log_loss_train.append(log_loss(y_train, 1 / (1 + np.exp(-y_t_p))))

        y_test_pred = gbclf.staged_decision_function(X_test)
        log_loss_test = []
        for y_t_p in y_test_pred:
            log_loss_test.append(log_loss(y_test, 1 / (1 + np.exp(-y_t_p))))

        # Min log-loss and the corresponding iteration
        log_loss_train_min_ind = np.argmin(log_loss_train) + 1
        log_loss_test_min_ind = np.argmin(log_loss_test) + 1
        log_loss_train_min = np.min(log_loss_train)
        log_loss_test_min = np.min(log_loss_test)
        min_iterations_train.append((log_loss_train_min, log_loss_train_min_ind))
        min_iterations_test.append((log_loss_test_min, log_loss_test_min_ind))

        # Plot the errors for both TRAIN and TEST sets (w/ the curr Learning Rate)
        plt.figure('GB learning rate: ' + str(lr))
        plt.plot(log_loss_test, 'r', linewidth=2)
        plt.plot(log_loss_train, 'g', linewidth=2)
        plt.legend(['log_loss_test', 'log_loss_train'])
        plt.draw()


    # Optimal TEST iteration for the learning rate 0.2
    print('Optimal iterations TEST vs. learning rate:')
    for t in zip(min_iterations_test, learning_rates):
        print('min: ', t[0][0], 'min_ind: ', t[0][1], 'learning rate: ', t[1])
    t = [(x[0], x[1]) for x, y in zip(min_iterations_test, learning_rates) if y == 0.2]
    opt_log_loss = t[0][0]
    opt_log_loss_ind = t[0][1]
    writefile('%0.2f %d' % (opt_log_loss, opt_log_loss_ind), 'log-loss-0.2.out')


    # ------------------------------------------------------
    # Deal with Random Forests
    # ------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=opt_log_loss_ind, random_state=241)
    clf.fit(X_train, y_train)
    y_test_pred_rf = clf.predict_proba(X_test)
    log_loss_test_rf = log_loss(y_test, y_test_pred_rf)
    # log-loss over the test set using Random Forests
    writefile('%0.2f' % (log_loss_test_rf), 'log-loss-rf.out')


    return 0


def main():
    # ------------------------------------------------------
    # Part 1. Random Forests
    # ------------------------------------------------------
    # fin = 'abalone.csv'
    # data = readdatawithheader(fin)
    # print('Random Forest task')
    # t_optimal = rf_for_abalone_age(data, RF_EPSILON=0.52)
    # s = '%d' % (t_optimal)
    # writefile(s, fin + ".out")

    # ------------------------------------------------------
    # Part 2. Gradient Boosting over Decision Trees
    # ------------------------------------------------------
    fin = 'gbm-data.csv'
    data = readdatawithheader(fin)
    print('Gradient Boosting task')
    gb(data)

    plt.show()
    print('Completed')


if __name__ == '__main__':
    main()
