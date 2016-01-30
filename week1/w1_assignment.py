import pandas
import re
from sphinx.util import matching


def readcsv(fin):
    return pandas.read_csv(fin, index_col='PassengerId')


def sex_count(data, fout_name):
    # # my submission:
    # male = data[data['Sex'] == 'male']['Sex'].count()
    # female = data[data['Sex'] == 'female']['Sex'].count()
    # sexnum = {}
    # sexnum['male'] = male
    # sexnum['female'] = female
    # return sexnum

    sex = data['Sex'].value_counts()

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%d %d' % (sex['male'], sex['female'])
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def survivors(data, fout_name):
    all = len(data.index)
    survived = len(data[data['Survived'] == 1].index)
    survived_percentage = float(survived) / all * 100

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%f' % (survived_percentage)
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def classPercentage(class_id, data, fout_name):
    all = len(data.index)
    people_in_class = len(data[data['Pclass'] == class_id].index)
    class_percentage = float(people_in_class) / all * 100

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%f' % (class_percentage)
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def ageStatistics(data, fout_name):
    ageSeries = data['Age']
    age_avg = ageSeries.mean()
    age_median = ageSeries.median()

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%f %f' % (age_avg, age_median)
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def corr(col1, col2, data, fout_name):
    c = data[[col1, col2]].corr()
    # print(c[col1][col2])

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%0.2f' % (c[col1][col2])
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def getName(s):
    # print (s)
    match = re.search('^(?P<last_name>.*),\s(?P<title>.*)[.]\s(?P<rest>.*)',
                      s,
                      re.I)
    if match is None:
        # print('-')
        return '-'
    else:
        # print('%s,%s,%s' % (match.group('last_name'), match.group('title'),match.group('rest')))
        match2 = re.search('\((?P<first_name>[^\s]*).*\)$',
                           match.group('rest'),
                           re.I)
        if match2 is None:
            # case we have no additional name
            match2 = re.search('(?P<first_name>[^\s]*).*$',
                               match.group('rest'),
                               re.I)
            # print(match2.group('first_name'))
            return match2.group('first_name')
        else:
            # case we have additional name and parse match2 in the same way
            # print(match2.group('first_name'))
            return match2.group('first_name')


def mostPopularName(sex, data, fout_name):
    names = data[data['Sex'] == sex]['Name']
    new_names = names.apply(getName)

    # fout = open(fout_name, 'w')
    # new_names.to_csv(fout)
    # fout.close()

    # Print out the results
    fout = open(fout_name, 'w')
    strout = '%s' % (new_names.mode().values[0])
    print(strout)
    print(strout, file=fout, end="")
    fout.close()


def main():
    fin = 'titanic.csv'
    data = readcsv(fin)

    print('Sex Count task...')
    # sex_count(data, 'sex_count.txt')

    print('Survived Count task...')
    # survivors(data, 'survivors_count.txt')

    print('First Class Passengers Count task...')
    # classPercentage(1, data, '1_class_passengers_count.txt')

    print('Age AVG and Median task...')
    # ageStatistics(data, 'age_statistics.txt')

    print('Correlation task...')
    # corr('SibSp', 'Parch', data, 'corr_SibSp_Parch.txt')

    print('Most popular name task...')
    mostPopularName('female', data, 'mostPopularName_female.txt')

    print('Completed!')


if __name__ == '__main__':
    main()
