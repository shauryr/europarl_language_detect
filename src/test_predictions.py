import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from tqdm import tqdm

test_data_path = '/data/szr207/europarl.test'

file_test = open(test_data_path,'r')
len_data = 400
space_letter = 0
x_test = []
y_test = []

for i in file_test:
	label, sentence = i.split('\t')
	x_test.append(sentence.rstrip())
	y_test.append(label)
x_data = []
for x in tqdm(x_test):
    x_row = np.ones(len_data, dtype=int) * space_letter
    for xi, i in zip(list(str(x)), np.arange(len_data)):
        x_row[i] = ord(xi)
    x_data.append(x_row)

x_test = np.array(x_data)

df = pd.DataFrame(y_test, columns=['label'])

y_data = pd.factorize(df['label'],sort=True)
labels = y_data[1]
y_data = y_data[0]
y_test = y_data

print labels

bst = xgb.Booster({'nthread': 4})
bst.load_model('basic.model')
dvalid = xgb.DMatrix(x_test, label=y_test)

y_pred = bst.predict(dvalid)

print(metrics.classification_report(y_test,y_pred,target_names=labels))


