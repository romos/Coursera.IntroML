from sklearn import cross_validation
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import metrics
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

import pandas
import numpy as np

__author__ = 'oderor'


def readdata(fin):
    return pandas.read_csv(fin, header=None)


def writefile(str, filename):
    fout = open(filename, 'w')
    print(str, file=fout, end="")
    fout.close()


def kNN_with_crossvalidation(data, normalize=False):
    K_MIN = 1
    K_MAX = 50
    K_STEP = 1
    FOLDS = 5

    K_OPTIMAL = K_MIN
    SCORE_OPTIMAL = 0

    X = data.iloc[:, 1:].values
    y = data.iloc[:, :1].values.ravel()
    N = len(y)

    if normalize:
        # X = [[1,2,0.5,4],[2,4,1,8],[3,6,1.5,16],[1,2,4,8],[1,2,4,8],[1,2,4,8]]
        X = preprocessing.scale(X)
        # print(X, X.mean(axis=0),X.std(axis=0))
        # print('Normalized:',X)

    print(X)

    # Create cross_validation sets (aka folds)
    kf = cross_validation.KFold(N, n_folds=FOLDS, shuffle=True, random_state=42)

    for k in np.arange(K_MIN, K_MAX + 1, K_STEP):
        # Init classifier
        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k)

        # Compute cross_validation classification and accuracy
        scores = cross_validation.cross_val_score(estimator=clf_knn, X=X, y=y, cv=kf, scoring='accuracy')

        avgscore = np.mean(scores)
        print('k: %d\t| avg score: %f\t| %s' % (k, avgscore, str(scores)))

        if avgscore > SCORE_OPTIMAL:
            SCORE_OPTIMAL = avgscore
            K_OPTIMAL = k

    return (K_OPTIMAL, SCORE_OPTIMAL)


def kNN_Regression_metrics_fitting_with_crossvalidation(data, normalize=False):
    K_MIN = 1
    K_MAX = 50
    K_STEP = 1
    P_MIN = 1.0
    P_MAX = 10.0
    P_COUNT = 200
    N_NEIGHBORS = 5
    METRICS = 'minkowski'
    WEIGHTS = 'distance'
    SCORING = 'mean_squared_error'
    FOLDS = 5

    P_OPTIMAL = P_MIN
    SCORE_OPTIMAL = 0

    boston = load_boston()
    X = boston.data
    y = boston.target
    N = np.size(y)

    if normalize:
        X = preprocessing.scale(X)

    # Create cross_validation sets (aka folds)
    kf = cross_validation.KFold(N, n_folds=FOLDS, shuffle=True, random_state=42)

    for p in np.linspace(P_MIN, P_MAX, P_COUNT):
        # Init classifier
        # print(p)
        # continue

        clf_knn_regr = neighbors.KNeighborsRegressor(n_neighbors=N_NEIGHBORS, metric=METRICS, p=p, weights=WEIGHTS)

        # Compute cross_validation classification and accuracy
        scores = cross_validation.cross_val_score(estimator=clf_knn_regr, X=X, y=y, cv=kf, scoring=SCORING)

        avgscore = np.mean(scores)
        print('p: %f\t| avg score: %f\t| %s' % (p, avgscore, str(scores)))

        if avgscore > SCORE_OPTIMAL:
            SCORE_OPTIMAL = avgscore
            P_OPTIMAL = p

    return (P_OPTIMAL, SCORE_OPTIMAL)


def perceptron_classifier(data_train, data_test):
    # Load train and test data sets
    X_train = data_train.iloc[:, 1:].values
    y_train = data_train.iloc[:, :1].values.ravel()
    X_test = data_test.iloc[:, 1:].values
    y_test = data_test.iloc[:, :1].values.ravel()

    # Init Perceptron
    clf = Perceptron(random_state=241)

    # --- Perceptron w/o normalization of Training Data Set ---

    # Fit Perceptron linear model using training data
    clf.fit(X_train, y_train)
    # Use the model to predict test data
    y_test_prediction = clf.predict(X_test)
    # Calculate accuracy:
    accuracy_notnorm = metrics.accuracy_score(y_test, y_test_prediction)

    # --- Perceptron w/ normalization of Training Data Set ---

    # feature scaling (standardization/normalization)
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Fit Perceptron using Training Set and predict results for tTest Set
    clf.fit(X_train_scaled, y_train)
    y_test_prediction = clf.predict(X_test_scaled)
    accuracy_norm = metrics.accuracy_score(y_test, y_test_prediction)

    # Note [FEATURE SCALING]:
    #   You MUST use fit_transform() over Training Set only.
    #   The scaler will compute necessary statistics like std_dev and mean [aka 'fit']
    #   and normalize Training Set [aka 'transform']
    #   But for the Test Set you must not fit the scaler again!
    #   Just re-use existing statistics and normalize the Test Set using transform() w/o fitting.

    print('Accuracy (non-normalized):', accuracy_notnorm)
    print('Accuracy (normalized):', accuracy_norm)
    diff = accuracy_norm - accuracy_notnorm
    print('Diff:', diff)

    return diff


def main():
    # --- Part 1 ---
    fin = 'wine.data'
    data = readdata(fin)
    print('kNN task...')
    (k, s) = kNN_with_crossvalidation(data)
    writefile('%d' % (k), 'wine.1.txt')
    writefile('%f' % (s), 'wine.2.txt')
    print('kNN task (normalized)...')
    (k, s) = kNN_with_crossvalidation(data, normalize=True)
    writefile('%d' % (k), 'wine.3.txt')
    writefile('%f' % (s), 'wine.4.txt')
    print('kNN Regression task (normalized)...')
    (p, s) = kNN_Regression_metrics_fitting_with_crossvalidation(data, normalize=True)
    writefile('%f' % (p), 'boston.p.txt')
    writefile('%f' % (s), 'boston.s.txt')

    # --- Part 2 ---
    in_train = 'perceptron-train.csv'
    in_test = 'perceptron-test.csv'
    data_train = readdata(in_train)
    data_test = readdata(in_test)
    print('Perceptron task...')
    accuracy_diff = perceptron_classifier(data_train, data_test)
    writefile('%0.3f' % (accuracy_diff),'perceptron.txt')

    print('Completed')


if __name__ == '__main__':
    main()
