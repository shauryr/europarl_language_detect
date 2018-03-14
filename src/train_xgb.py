import glob
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle
import gc
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn import metrics


my_dir = '/data/szr207/txt/'
len_data = 400
space_letter = 0
iterations = 400
dict_language_id = {}
folder_language = glob.glob(my_dir + '*')
dict_language_files = {}
dict_language_text = defaultdict(list)
np_x_data = []
np_y_data = []

for language in folder_language:
    dict_language_files[language] = glob.glob(language + '/*')

# print dict_language_files[folder_language[0]][0].split('\\')[-2]
# print dict_language_files[folder_language[0]][0].split('\\')[-1]

for language in folder_language:
    list_files = dict_language_files[language]
    for files in list_files:
        with open(files, 'r') as myfile:
            data = myfile.read().replace('\n', '')
            dict_language_text[language.split('/')[-1]].append(data[18:len_data+18])
    print language.split('/')[-1], len(dict_language_text[language.split('/')[-1]])

for language in tqdm(dict_language_text):
    for text in dict_language_text[language]:
        np_x_data.append(text)
        np_y_data.append(language)

dict_language_text = None
dict_language_files = None
x_data = []
for x in tqdm(np_x_data):
    x_row = np.ones(len_data, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(len_data)):
        x_row[i] = ord(xi)
    x_data.append(x_row)

df = pd.DataFrame(np_y_data, columns=['label'])
count = 0
for language in dict_language_id:
    dict_language_id[language] = count
    count += 1

y_data = pd.factorize(df['label'],sort=True)
labels = y_data[1]
y_data = y_data[0]

x_data = np.array(x_data)
#y_data = np.array(np_y_data)

print('Total number of samples:', len(x_data))
#print('Use: ', max_data_size)
# x_data = np.array(x_data)
# y_data = np.array(y_data)

print('x_data sample:')
print(x_data[0])
print('y_data sample:')
print(y_data[0])
print('labels:')
print(labels)

x_train = x_data
y_train = y_data
gc.collect()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      test_size=0.1, random_state=2017)
gc.collect()
num_class = len(labels)
dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_valid, label=y_valid)
watchlist = [(dvalid, 'valid'), (dtrain, 'train')]

param = {'objective': 'multi:softmax',
         'eta': '0.3', 'max_depth': 10,
         'silent': 1, 'nthread': 64,
         'num_class': num_class,
         'eval_metric': 'merror'}

model = xgb.train(param, dtrain, iterations, watchlist, early_stopping_rounds=20,
                  verbose_eval=5)
gc.collect()

pred = model.predict(dvalid)

print(metrics.classification_report(y_valid,pred,target_names=labels))

model.save_model('basic.model')
#bst = xgb.Booster({'nthread': 4})
#bst.load_model('basic.model')


