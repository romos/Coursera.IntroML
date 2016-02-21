import pandas
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA

__author__ = 'oderor'


# File header fields:
FULLDESCRIPTION = 'FullDescription'
LOCATIONNORMALIZED = 'LocationNormalized'
CONTRACTTIME = 'ContractTime'
SALARYNORMALIZED = 'SalaryNormalized'


def readdata(fin):
    return pandas.read_csv(fin, header=None)


def readdatawithheader(fin):
    return pandas.read_csv(fin)


def writefile(str, filename):
    fout = open(filename, 'w')
    print(filename, ':', str)
    print(str, file=fout, end="")
    fout.close()


def toLowerCase(data, textcolumn):
    data[textcolumn] = data[textcolumn].apply(lambda x: x.lower())
    return


def ridge(data_train, data_test):
    vectorizer = TfidfVectorizer(min_df=5)
    enc = DictVectorizer()

    print("Processing Train Data Set...")
    # TRAIN SET
    # Text field to lower case and replace all special chars with spaces.
    toLowerCase(data_train, FULLDESCRIPTION)
    data_train[FULLDESCRIPTION] = data_train[FULLDESCRIPTION].replace('[^a-zA-Z0-9]', ' ', regex=True)
    # Create feature vectors based on texts
    # *only for the words that exist min in 5 docs
    description_vectors_train = vectorizer.fit_transform(data_train[FULLDESCRIPTION])
    # insert unique NaN values
    data_train[LOCATIONNORMALIZED].fillna('nan', inplace=True)
    data_train[CONTRACTTIME].fillna('nan', inplace=True)
    # one-hot coding for 'Location' and 'Contract' features (their values are from a finite predefined range))
    X_train_categ = enc.fit_transform(data_train[[LOCATIONNORMALIZED, CONTRACTTIME]].to_dict('records'))

    print("Processing Test Data Set...")
    # TEST SET
    # Text field to lower case and replace all special chars with spaces.
    toLowerCase(data_test, FULLDESCRIPTION)
    data_test[FULLDESCRIPTION] = data_test[FULLDESCRIPTION].replace('[^a-zA-Z0-9]', ' ', regex=True)
    # Create feature vectors based on texts
    # *only for the words that exist min in 5 docs
    description_vectors_test = vectorizer.transform(data_test[FULLDESCRIPTION])
    # insert unique NaN values
    data_test[LOCATIONNORMALIZED].fillna('nan', inplace=True)
    data_test[CONTRACTTIME].fillna('nan', inplace=True)
    # one-hot coding for 'Location' and 'Contract' features (their values are from a finite predefined range))
    # enc = DictVectorizer()
    X_test_categ = enc.transform(data_test[[LOCATIONNORMALIZED, CONTRACTTIME]].to_dict('records'))

    # Form preprocessed datasets
    X_train = hstack([description_vectors_train, X_train_categ])
    y_train = data_train[SALARYNORMALIZED]
    X_test = hstack([description_vectors_test, X_test_categ])

    # Fit linear regression:
    print("Fitting Ridge classifier using train data...")
    clf = Ridge(alpha=1)
    clf.fit(X_train, y_train)

    print("Predictions:")
    y_predict = clf.predict(X_test)
    print(y_predict)
    return y_predict


def pca(data_prices, data_djia):
    prices = data_prices.drop('date', axis=1)
    nd_djia = data_djia['^DJI'].values

    # ------------------------------------------------------
    # Fit PCA and find the components to keep 90% dispersion
    # ------------------------------------------------------
    # # 1 way:
    # #     in this case PCA will automatically fit up to 90% dispersion
    # #     due to the N_COMPONENTS parameter being between 0 and 1.
    # N_COMPONENTS = 0.9
    # pca = PCA(n_components=N_COMPONENTS)
    # pca.fit(data)
    # componentsToKeep90Dispersion = pca.n_components_
    # # 2 way:
    # #     Fit PCA with some N_COMPONENTS value
    # #     and then manually calculate how many components will be enough
    # #     to fit 90% dispersion
    N_COMPONENTS = 10
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(prices)
    sum = 0
    count = 0
    for i in pca.explained_variance_ratio_:
        sum += i
        count += 1
        if sum > 0.9:
            break
    componentsToKeep90Dispersion = count

    # ------------------------------------------------------
    # Apply PCA to the initial price data set and get the first component
    # ------------------------------------------------------
    prices_pca = pca.transform(prices)
    firstComponent_pca = prices_pca[:, 0]

    # calculate Pearson correlation between the first component and DJIA
    corr_DJI_PCA0 = np.corrcoef(firstComponent_pca, nd_djia)[0, 1]

    # print(prices.shape, prices)
    # print(pca.components_.shape, pca.components_)

    t = zip(pca.components_[0], prices.columns)
    mostValuableCompanyTuple = max(t)

    return componentsToKeep90Dispersion,\
           corr_DJI_PCA0,\
           mostValuableCompanyTuple[1]


def main():
    # # --- Part 1. Ridge regression ---
    # fin = 'salary-train.csv'
    # data_train = readdatawithheader(fin)
    # fin = 'salary-test-mini.csv'
    # data_test = readdatawithheader(fin)
    # print('Ridge task')
    # result = ridge(data_train, data_test)
    # s = ' '.join('%0.2f' % y for y in result)
    # writefile(s, (fin + ".out"))

    # --- Part 2. Principle Component Analysis aka PCA ---
    fin = 'close_prices.csv'
    data_prices = readdatawithheader(fin)
    fin = 'djia_index.csv'
    data_djia = readdatawithheader(fin)
    print('PCA task...')
    componentsToKeep90Dispersion, corr_DJI_PCA0, mostValuableCompany = \
        pca(data_prices, data_djia)

    s = '%d' % (componentsToKeep90Dispersion)
    writefile(s, ("pca90dispersion.out"))
    s = '%0.2f' % (corr_DJI_PCA0)
    writefile(s, ("corr_DJI_PCA0.out"))
    s = '%s' % (mostValuableCompany)
    writefile(s, ("mostValuableCompany.out"))

    print('Completed')


if __name__ == '__main__':
    main()
