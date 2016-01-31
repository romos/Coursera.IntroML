__author__ = 'oderor'

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def readcsv(fin):
    return pandas.read_csv(fin, index_col='PassengerId')


def myDecisionTree(X, Y):
    print('Initializing DecisionTreeClassifier...')
    clf = DecisionTreeClassifier(random_state=241)
    print('Fitting...')
    clf.fit(X, Y)
    print('Success!')

    importances = clf.feature_importances_

    return importances


def main():
    fin = 'titanic.csv'
    data = readcsv(fin)
    # get only necessary data
    subData = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
    # drop NaN values
    subData = subData.dropna()
    # define X and Y datasets for Decision Tree Classification
    dataX = subData[['Pclass', 'Fare', 'Age', 'Sex']]
    dataY = subData['Survived']
    # map {'male' -> 1, 'female' -> 0}
    dataX.loc[dataX['Sex'] == 'female', 'Sex'] = 0
    dataX.loc[dataX['Sex'] == 'male', 'Sex'] = 1

    # dataX['Sex'] = dataX['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    # dataX['Sex'] = dataX['Sex'].map({'female': 0, 'male': 1})

    print('myDecisionTree task...')
    importances = myDecisionTree(dataX.values, dataY.values)
    series_importances = pandas.Series(importances, index=dataX.columns).sort_values(ascending=False)

    # Print out the results
    fout = open('dt.txt', 'w')
    strout = ' '.join(series_importances[:2].index.values)
    print(strout)
    print(strout, file=fout, end="")
    fout.close()

    print('Completed')


if __name__ == '__main__':
    main()
