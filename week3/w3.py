import math
import numpy as np
import pandas
import operator

from sklearn.svm import SVC
from sklearn import datasets
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score as ras
from sklearn.metrics import\
    accuracy_score,\
    precision_score,\
    recall_score,\
    f1_score,\
    precision_recall_curve

__author__ = 'oderor'


def readdata(fin):
    return pandas.read_csv(fin, header=None)


def readdatawithheader(fin):
    return pandas.read_csv(fin)


def writefile(str, filename):
    fout = open(filename, 'w')
    print(filename, ':', str)
    print(str, file=fout, end="")
    fout.close()


def svm_data(data, c=100000, random_state=241):
    X = data.iloc[:, 1:].values
    y = data.iloc[:, :1].values.ravel()
    clf = SVC(C=c, random_state=random_state)
    clf.fit(X, y)
    return clf.support_


def svm_text():
    newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
    )
    # print("newsgroups.target.shape:", newsgroups.target.shape)
    # print("newsgroups.filenames.shape:", newsgroups.filenames.shape)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroups.data)

    print("Fitting cross-validation SVC...")

    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = cross_validation.KFold(newsgroups.filenames.size,
                                n_folds=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(vectors, newsgroups.target)

    A_OPTIMAL = gs.grid_scores_[0]
    for a in gs.grid_scores_:
        if a.mean_validation_score > A_OPTIMAL.mean_validation_score:
            A_OPTIMAL = a
    print("Optimal mean validation score:", A_OPTIMAL.mean_validation_score)
    print("Optimal parameter C:", A_OPTIMAL.parameters)

    print("Fitting SVC using optimal parameters...")

    clf = SVC(C=A_OPTIMAL.parameters['C'], kernel='linear')
    clf.fit(vectors, newsgroups.target)

    print("Finding 10 most valuable words...")


    # transform sparse matrix into a dense array & get absolute values
    coef_abs = np.absolute(clf.coef_[0:].toarray())
    print('coef_abs:', type(coef_abs), coef_abs.shape)

    # get indices of the largest 10 elements
    coef_inds = np.argsort(coef_abs)
    print('coefs_inds:', type(coef_inds), coef_inds.shape, coef_inds)

    coef_inds10 = coef_inds[:, -10:]
    print('coefs_inds10:', type(coef_inds10), coef_inds10.shape, coef_inds10)

    allwords = np.array(vectorizer.get_feature_names())
    print('all vectorizer words:', type(allwords), allwords.shape, allwords)

    topwords = allwords[coef_inds10]
    print('TOP 10 WORDS:', topwords)

    topwords = np.sort(topwords)
    print(type(topwords), topwords.shape, topwords)

    s = ','.join(topwords[0, :])

    return s


def w_next(w, k, l, y, X, C=0):
    w1 = w[0]
    w2 = w[1]

    s1 = 0
    s2 = 0
    for i in range(l):
        a = w1 * X[i, 0] + w2 * X[i, 1]
        a = 1 + math.exp(-1 * y[i] * a)
        a = 1 - 1 / a
        # s1 = s1 + y[i]*X[0][i]*a
        # s2 = s2 + y[i]*X[1][i]*a
        s1 = s1 + y[i] * X[i, 0] * a
        s2 = s2 + y[i] * X[i, 1] * a
    w1 = w1 + k / l * s1 - k * C * w1
    w2 = w2 + k / l * s2 - k * C * w2

    w_res = np.array([w1, w2])
    return w_res


def gd(W_INIT, k, l, y, X, C=0, EPSILON=0.00001, ITER_MAX=10000):
    w = W_INIT
    iter = 0
    while True:
        w_old = np.array([w[0], w[1]])
        w = w_next(w, k, l, y, X, C)
        e = np.linalg.norm(w - w_old)
        iter += 1
        print('%d: %0.10f' % (iter, e))
        if e <= EPSILON or iter >= ITER_MAX:
            break
    return w


def sigmoid(x, w):
    return 1 / (1 + math.exp((-1) * (w.dot(x))))


def auc_roc(y_orig, x_orig, w):
    y_predicted = np.dot(x_orig, w.reshape(w.size, 1))
    return ras(y_orig, y_predicted)


def logregression(data):
    X = data.iloc[:, 1:]
    y = data.iloc[:, :1]  # .values.ravel()
    l = len(y)

    k = 0.1
    l2reg = 10
    W_INIT = np.zeros((X.shape[1]))

    # GD w/o regularization
    print('GD w/o regularization')
    w_noreg = gd(W_INIT, k, l, y.values.ravel(), X.values)
    res_noreg = auc_roc(y.values.ravel(), X.values, w_noreg)

    # GD w/ L2 regularization
    print('GD w/ L2 regularization')
    w_l2reg = gd(W_INIT, k, l, y.values.ravel(), X.values, C=l2reg)
    res_l2reg = auc_roc(y.values.ravel(), X.values, w_l2reg)

    print('w/o regularization:', res_noreg)
    print('l2  regularization:', res_l2reg)

    return (res_noreg, res_l2reg)


def errorMatrix(data):
    y = data
    # y = data.iloc[:10, :]
    cols = ['TP', 'FP', 'FN', 'TN']
    tp = len(y[(y['true'] == 1) & (y['pred'] == 1)])
    fp = len(y[(y['true'] == 0) & (y['pred'] == 1)])
    fn = len(y[(y['true'] == 1) & (y['pred'] == 0)])
    tn = len(y[(y['true'] == 0) & (y['pred'] == 0)])
    errorMatrix = pandas.Series([tp, fp, fn, tn], index=cols)

    cols = ['accuracy', 'precision', 'recall', 'f1']
    metrics = pandas.Series([accuracy_score(y['true'], y['pred']),
                             precision_score(y['true'], y['pred']),
                             recall_score(y['true'], y['pred']),
                             f1_score(y['true'], y['pred'])],
                            index=cols)
    return errorMatrix, metrics


def roc_scores(data):
    y = data
    rocs = pandas.Series()
    for c in y.columns.values[1:]:
        temp1 = ras(y['true'], y[c])
        rocs.set_value(c,temp1)
        # temp = pandas.Series([temp1], index=[c])
        # rocs = rocs.append(temp) # also works fine
    return rocs


def prcurve(data):
    y = data
    dict = {}
    for c in y.columns.values[1:]:
        precision, recall, thresholds = precision_recall_curve(y['true'], y[c])
        ps = pandas.Series(precision)
        rs = pandas.Series(recall)
        df = pandas.DataFrame({'precision': ps, 'recall': rs})
        dict[c] = df
    return dict

def prcurve_filter(dict, recall_threshold=0.7):
    new_dict = {}
    for k,v in dict.items():
        df = dict[k]
        new_dict[k] = df[df['recall'] >= recall_threshold]['precision'].max()

    return new_dict

def main():
    # --- Part 1, SVM data ---
    fin = 'svm-data.csv'
    data = readdata(fin)
    print('SVM data task...')
    svindices = svm_data(data, c=100000, random_state=241)
    svindices = svindices+1
    s = ','.join(str(v) for v in svindices)
    writefile(s, (fin + ".out"))

    # --- Part 2, SVM text ---
    print('SVM text task...')
    s = svm_text()
    writefile(s, ("svm_text.out"))

    # --- Part 3, Log Reg ---
    fin = 'data-logistic.csv'
    data = readdata(fin)
    print('Log Reg task...')
    (res_noreg, res_l2reg) = logregression(data)
    s = '%0.3f %0.3f' % (res_noreg, res_l2reg)
    writefile(s, (fin + ".out"))

    # --- Part 4, Metrics ---
    print('Metrics task...')

    fin = 'classification.csv'
    data = readdatawithheader(fin)

    print('2. Error Matrix')
    em, metrics = errorMatrix(data)
    print(em)
    s = '%d %d %d %d' % \
        (em['TP'], em['FP'], em['FN'], em['TN'])
    writefile(s, (fin + ".errmatrix.out"))

    print('3. Error Matrix Metrics')
    print(metrics)
    s = '%0.2f %0.2f %0.2f %0.2f' % \
        (metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'])
    writefile(s, (fin + ".metrics.out"))

    fin = 'scores.csv'
    data = readdatawithheader(fin)

    print('5. AUC-ROC')
    rocs = roc_scores(data)
    print(rocs)
    maxroc = rocs.idxmax()
    s = '%s' % (maxroc)
    writefile(s, (fin + ".auc-roc-max.out"))

    print('6. PR cuve')
    pr_dict = prcurve(data)
    pr_maxs = prcurve_filter(pr_dict,recall_threshold=0.7)
    print(pr_maxs)
    maxPrecisionClassifier = max(pr_maxs.items())[0]
    s = '%s' % (maxPrecisionClassifier)
    writefile(s, (fin + ".max-precision.out"))

    print('Completed')


if __name__ == '__main__':
    main()
