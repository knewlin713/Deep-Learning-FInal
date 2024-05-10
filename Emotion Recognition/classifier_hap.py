from re import I
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import joblib
data = pd.read_pickle('./working_data/extracted_embeddings.pk')

data_filtered = []
for row, v in data.iterrows():
    #  {'happy':1,'sad':2,'neutral':3,'angry':4,'other':10}
    #  {negative':1,'neutral':2, positve: 0}
    if v.labels != 'discard' and (v.labels == 1 or v.labels == 2 or v.labels==5 or v.labels==3):

        if (v.labels == 2 or v.labels == 5 or v.labels == 3):
            d = v
            d.labels = 0
            data_filtered.append(d)
        else:
            data_filtered.append(v)
print(len(data_filtered)/len(data), '% valid data ---', len(data_filtered), 'samples')

embeddings = []
subjects = []
labels = []
for em in data_filtered:
    embeddings.append(em['dt'])
    subjects.append(em['subject'])
    labels.append(em['labels'])


X = np.array(embeddings)
X_sub = np.array(subjects)
y = np.array(labels)


X_prep = []
for e in X:
    X_prep.append(np.mean(e,axis=0))
X = np.array(X_prep)


print(X.shape)
possible_subjects = list(range(1001, 1092))
TRAIN_SIZE = 0.8 
TEST_SIZE = 0.2 
RANDOM_STATE = 600
X_train_subs, X_test_subs = train_test_split(possible_subjects, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(X.shape[0]):
    if data['subject'][i] in X_train_subs:
        X_train.append(X[i])
        y_train.append(y[i])
    elif data['subject'][i] in X_test_subs:
        X_test.append(X[i])
        y_test.append(y[i])


X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


from imblearn.datasets import make_imbalance
X_train, y_train = make_imbalance(X_train, y_train, sampling_strategy={0: 720, 1: 720},random_state=RANDOM_STATE)
X_test, y_test = make_imbalance(X_test, y_test, sampling_strategy={0: 320, 1: 320},random_state=RANDOM_STATE)



model = SVC(random_state=99999, kernel='rbf', gamma=0.1, C=1000, probability=True).fit(X_train, y_train)


from sklearn.metrics import f1_score
print('weighted f-score', f1_score(y_test, model.predict(X_test), average='weighted'))
print('Accuracy', model.score(X_test, y_test))
conf_mat = confusion_matrix(y_test, model.predict(X_test))
print(conf_mat)

plt.figure()
import seaborn as sns
sns.heatmap(conf_mat, annot=True)
plt.show()

