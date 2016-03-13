
# coding: utf-8

# Импорт данных

# In[1]:

import pandas
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import time
import datetime
from sklearn.metrics import roc_auc_score


# In[2]:

features_orig = pandas.read_csv('../data/features.csv',index_col='match_id')
features_test_orig = pandas.read_csv('../data/features_test.csv',index_col='match_id')


# Удаляем признаки, связанные с итогами матча 

# In[3]:

features = features_orig
features_test = features_test_orig

for c in ['duration',
#           'radiant_win',
          'tower_status_radiant',
          'tower_status_dire',
          'barracks_status_radiant',
          'barracks_status_dire']:
    if c in features.columns:
        features = features.drop(c,axis=1)
    if c in features_test.columns:
        features_test = features_test.drop(c,axis=1)


# Ищем пропуски среди оставшихся признаков

# In[4]:

features.count()[features.count() < features.shape[0]]


# Получаем 12 фич
# 
# 4 относятся к событию first blood:  
# 
# **first_blood_time               77677**  
# время самого первого убийства соперника (независимо от команды). Может не произойти за первые 5 минут, поэтому может быть пропущено в выборке.  
# **first_blood_team               77677**  
# команда, игрок который совершил first blood.  
# **first_blood_player1            77677**  
# **first_blood_player2            53243**  
# пара игроков, причастных к first blood  
# 
# 8 событий относятся к каждой команде в отдельности.
# Если они не попадают в первые 5 минут, то в выборке образуется пропуск.
# 
# **radiant_bottle_time            81539**  
# время покупки предмета 'bottle', который позволяет восстанавливать здоровье и ману Героев, а также хранить руны  
# **radiant_courier_time           96538**  
# время приборетения "courier" - предмета, позволяющего транспортировать предметы  
# **radiant_flying_courier_time    69751**  
# время приборетения "flying_courier" - предмета, позволяющего транспортировать предметы. Особая разновидность flying  
# **radiant_first_ward_time        95394**  
# время установки командой первого "наблюдателя", т.е. предмета, который позволяет видеть часть игрового поля  
# 
# Аналогичные характеристики логируются и для второй команды:  
# **dire_bottle_time               81087**  
# **dire_courier_time              96554**  
# **dire_flying_courier_time       71132**  
# **dire_first_ward_time           95404**  
# 
# Теоретически, существует возможность и того, что соответствующие покупки предметов или события вообще не имели места, но это практически невозможно с учетом особенностей игры Dota 2.

# Обрабатываем пропуски в данных

# In[5]:

# Заменить пропуски на нули - самый простой вариант.
# Рекомендован при использовании логистической регрессии,
# т.к. пропущенное значение перестает влиять на предсказание. 
features = features.fillna(0)
features_test = features_test.fillna(0)

### Можно пробовать заменить на очень большое или очень маленькое значение.
### Это полезно для деревьев: благодаря этому получается отнести объекты с пропусками в отдельную ветвь дерева
# features = features.fillna(999999999)
# features_test = features_test.fillna(999999999)

### Можно пробовать заменить на среднее значение столбца.
# features = features.fillna(features.mean())
# features_test = features_test.fillna(features_test.mean())


# Цель - предсказать значение признака **radiant_win**  
# 0 - если победила команда Dire  
# 1 - если победила команда Radiant  

# In[6]:

X = features.iloc[:, :-1].values
y = features.iloc[:, -1:].values.ravel()
N = len(y)

X_test = features_test.iloc[:, :].values


# In[7]:

cvkfold = cross_validation.KFold(N,
                           n_folds=5,
                           shuffle=True,
                           random_state=241)


# In[8]:

#
# Можно сделать подбор нужного кол-ва деревьев с помощью cross_val_score и передать туда метрику.
#
# print("Gradient Boosting. CrossValScore")
# trees = [10,20,30]
# for t in trees:
#     gbclf = GradientBoostingClassifier(n_estimators=t,
#                                       # verbose=True,
#                                       random_state=241)
#     start_time = datetime.datetime.now()
#     cvscores = cross_validation.cross_val_score(gbclf, X, y, scoring='roc_auc', cv=cvkfold)
#     elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()   
#     avgcvscore = np.mean(cvscores)
#     print('Trees: %d\t Avg Score: %.5f\t Elapsed Time: %d' % (t, avgcvscore, elapsed_seconds))

#
# Можно "руками сделать все, что нужно для подсчет AUC_ROC:
#
print("Gradient Boosting. Manual CrossValScore")
trees = [10,20,30]
for t in trees:
    gbclf = GradientBoostingClassifier(n_estimators=t,
                                      # verbose=True,
                                      random_state=241)
    auc_roc_metrics = []
    
    start_time = datetime.datetime.now()
    
    for (train, test) in cvkfold:
        gbclf.fit(X[train], y[train])
        pred = gbclf.predict_proba(X[test])[:,1]
        auc_roc_metrics.append(roc_auc_score(y[test],pred))
    
    elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
    
    auc_roc_metrics_avg = np.mean(auc_roc_metrics)    
    print('Trees: %d\t Avg Score: %.5f\t Elapsed Time: %d' % (t, auc_roc_metrics_avg, elapsed_seconds))


# **Получается следующий результат:**  
# Trees: 10    Avg Score: 0.66439    Elapsed Time: 52  
# Trees: 20    Avg Score: 0.68285    Elapsed Time: 102  
# Trees: 30    Avg Score: 0.68950    Elapsed Time: 151  
#   
# Градиентный бустинг, конечно, может переобучаться, и, возможно, если позволяли бы ресурсы, можно чуть больше обучить его.  
# Однако результат не станет сильно лучше.  
#   
# Для улучшения работы бустинга можно порпобовать:  
# * подбирать число дереьев  
# * ограничить глубину деревьев и другие параметры
# * заполнять пропуски в данных не просто нулями, а оч.большими или маленькими значениями, чтобы соответствующие объекты скорее уходили в крайние ветви деревьев
# и т.п.
# 

# In[9]:

# t = 30
# gbclf = GradientBoostingClassifier(n_estimators=t,
#                                   verbose=True,
#                                   random_state=241)
# gbclf.fit(X,y)
# # pred = gbclf.predict_proba(X_test)
# pred = gbclf.predict(X_test)


# -------------------------------------------------------------------------------------------------------------------------------

# In[10]:

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_scl=scale.fit_transform(X)
X_test_scl = scale.transform(X_test)


# In[12]:

from sklearn.linear_model import LogisticRegression

print("Linear Regression w/L2 regularization")

Cs = np.power(10.0, np.arange(-5,6))
C_optimal = Cs[0]
AUC_ROC_optimal = 0
for C in Cs:
    lgclf = LogisticRegression(random_state=241, penalty='l2',C=C)
    auc_roc_metrics = []
    
    start_time = datetime.datetime.now()
    
    for (train, test) in cvkfold:
        lgclf.fit(X_scl[train], y[train])
        pred = lgclf.predict_proba(X_scl[test])[:,1]
        auc_roc_metrics.append(roc_auc_score(y[test],pred))
    
    elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
    
    auc_roc_metrics_avg = np.mean(auc_roc_metrics)    
    if auc_roc_metrics_avg > AUC_ROC_optimal:
        C_optimal = C
        AUC_ROC_optimal = auc_roc_metrics_avg

    print('C: %.10f\t Avg Score: %.10f\t Elapsed Time: %d' % (C, auc_roc_metrics_avg, elapsed_seconds))
    
print('Optimal C: ', C_optimal, 'Optimal AUC_ROC: ', AUC_ROC_optimal)


# Optimal C:  0.01 Optimal AUC_ROC:  0.716341468544
# 
# Таким образом, видно, что при C = 0.01 точность является максимальной (среди рассмотренных).  
# Скорость Лог.Рег. на порядок выше Град.Бустинга, а точность одновременно превосходит Град.Бустинг. 
# Линейные методы в общем случае больше пригодны для разреженных данных.

# Удалим категориальные признаки (те, которые принимают значения из конечного множества) и проверим, изменится ли качество Лог.Регрессии

# In[17]:

features_nocateg = features
features_test_nocateg = features_test

cols = []
for i in range(1,6):
    cols.append('r%d_hero' % (i))
    cols.append('d%d_hero' % (i))
cols.append('lobby_hero')

print(cols)

for c in cols:
    if c in features_nocateg.columns:
        features_nocateg = features_nocateg.drop(c,axis=1)
    if c in features_test_nocateg.columns:
        features_test_nocateg = features_test_nocateg.drop(c,axis=1)

X_nocateg = features_nocateg.iloc[:, :-1].values
y = features_nocateg.iloc[:, -1:].values.ravel()
N = len(y)

X_test_nocateg = features_test_nocateg.iloc[:, :].values

scale = StandardScaler()
X_nocateg_scl=scale.fit_transform(X_nocateg)
X_test_nocateg_scl = scale.transform(X_test_nocateg)


# In[19]:

print("Linear Regression w/L2 regularization")

Cs = np.power(10.0, np.arange(-5,6))
C_optimal = Cs[0]
AUC_ROC_optimal = 0
for C in Cs:
    lgclf = LogisticRegression(random_state=241, penalty='l2',C=C)
    auc_roc_metrics = []
    
    start_time = datetime.datetime.now()
    
    for (train, test) in cvkfold:
        lgclf.fit(X_nocateg_scl[train], y[train])
        pred = lgclf.predict_proba(X_nocateg_scl[test])[:,1]
        auc_roc_metrics.append(roc_auc_score(y[test],pred))
    
    elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
    
    auc_roc_metrics_avg = np.mean(auc_roc_metrics)    
    if auc_roc_metrics_avg > AUC_ROC_optimal:
        C_optimal = C
        AUC_ROC_optimal = auc_roc_metrics_avg

    print('C: %.10f\t Avg Score: %.10f\t Elapsed Time: %d' % (C, auc_roc_metrics_avg, elapsed_seconds))
    
print('Optimal C: ', C_optimal, 'Optimal AUC_ROC: ', AUC_ROC_optimal)


# С учетом того, что мы перестали учитывать категориальные признаки, качество практически не изменилось. И в предыдущем случае, и сейчас мы неправильно учитывали эти признаки - считали их простыми числовыми.  
# Стоило ввести мешок слов, чтобы сохранить их для алгоритма, но сделать их использование более разумным.
# 
# 
# 
# 

# А сколько всего существует различных героев в нашей оригинальной выборке?  
# Возьмем все столбцы с героями и найдем среди всех значений уникальные.

# In[53]:

cols = []
for i in range(1,6):
    cols.append('r%d_hero' % (i))
    cols.append('d%d_hero' % (i))

heroes = features[cols].values
unique_heroes = np.unique(heroes)
print(unique_heroes)

# unique_heroes_stat = pandas.Series(heroes.ravel(),index=range(0,len(heroes.ravel()))).value_counts()


# Можно заметить, что не все герои фигурируют в данных. Но в любом случае, максимум их тут 112.
# Теперь построим bag of words:

# In[92]:

# N — количество различных героев в выборке
N = np.max(unique_heroes)
X_pick = np.zeros((features.shape[0], N))
for i, match_id in enumerate(features.index):
    for p in range(5):
        X_pick[i, features[('r%d_hero' % (p+1))].iloc[i] - 1] = 1
        X_pick[i, features[('d%d_hero' % (p+1))].iloc[i] - 1] = -1
        
        
X_pick_test = np.zeros((features_test.shape[0], N))
for i, match_id in enumerate(features_test.index):
    for p in range(5):
        X_pick_test[i, features_test[('r%d_hero' % (p+1))].iloc[i] - 1] = 1
        X_pick_test[i, features_test[('d%d_hero' % (p+1))].iloc[i] - 1] = -1


# In[96]:

X_bag_scl = np.hstack((X_nocateg_scl, X_pick))
X_test_bag_scl = np.hstack((X_test_nocateg_scl, X_pick_test))
y = features_nocateg.iloc[:, -1:].values.ravel()


# Снова проведем лог.регрессию:

# In[102]:

print("Linear Regression w/L2 regularization")

Cs = np.power(10.0, np.arange(-5,6))
C_optimal = Cs[0]
AUC_ROC_optimal = 0
for C in Cs:
    lgclf = LogisticRegression(random_state=241, penalty='l2',C=C)
    auc_roc_metrics = []
    
    start_time = datetime.datetime.now()
    
    for (train, test) in cvkfold:
        lgclf.fit(X_bag_scl[train], y[train])
        pred = lgclf.predict_proba(X_bag_scl[test])[:,1]
        auc_roc_metrics.append(roc_auc_score(y[test],pred))
    
    elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()
    
    auc_roc_metrics_avg = np.mean(auc_roc_metrics)    
    if auc_roc_metrics_avg > AUC_ROC_optimal:
        C_optimal = C
        AUC_ROC_optimal = auc_roc_metrics_avg

    print('C: %.10f\t Avg Score: %.10f\t Elapsed Time: %d' % (C, auc_roc_metrics_avg, elapsed_seconds))
    
print('Optimal C: ', C_optimal, 'Optimal AUC_ROC: ', AUC_ROC_optimal)


# Видно, что теперь качество регрессии улучшилось.  
# Оптимальные настройки регрессии:  
# Optimal C:  0.1 Optimal AUC_ROC:  0.751919966491  
# Благодаря более корректному спользованию категориальных признаков в сочетании с нормализацией остальных числовых признаков мы получили более эффективный классификатор.

# Теперь протестируем на тестовых данных Логистич. Регрессию с параметром C=0.1  

# In[117]:

print("Linear Regression w/L2 regularization")

C = 0.1
lgclf = LogisticRegression(random_state=241, penalty='l2',C=C)
start_time = datetime.datetime.now()
lgclf.fit(X_bag_scl, y)
pred = lgclf.predict(X_test_bag_scl)
pred_prob = lgclf.predict_proba(X_test_bag_scl)[:,1]
elapsed_seconds = (datetime.datetime.now() - start_time).total_seconds()

print('C: %.10f\t Elapsed Time: %d' % (C, elapsed_seconds))


# Минимальные и максимальные вероятности, полученные на тестовой выборке:

# In[124]:

print('MAX prob:', np.max(pred_prob))
print('MIN prob:', np.min(pred_prob))

