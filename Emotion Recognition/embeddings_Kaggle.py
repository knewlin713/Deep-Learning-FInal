import numpy as np
import pandas as pd
import csv
from pandas import read_parquet
import os

path='Deep-Learning-FInal/Emotion Recognition/Data/Kaggle/set/TRAIN'
csv_file='Deep-Learning-FInal/Emotion Recognition/Data/Kaggle/set/TRAIN.csv'

ext_audio = '.wav'

file_path = []
file_label = [] #by this we mean pos, neu, or neg
csv_parse = []

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        words = line.split(',')
        words = [word.strip() for word in words]
        csv_parse.append(words)

for file in list:
    for file_digit in csv_parse:
        if file_digit[0] == str(file): #takes emotion and label based of finding file in csv
            file_label.append[file_digit[1]]
            file_path.append(path + file)
    

labels_df = pd.DataFrame(file_label, columns=['Labels'])
path_df = pd.DataFrame(file_path, columns=['Path'])

data_set = pd.concat([labels_df, None, path_df], axis=1)

data_set.to_parquet('./working_data/Kaggle.pq')
print('Successfully saved the DF to a parquet in ./working_data')

kaggle = read_parquet('./working_data/Kaggle.pq')

rep_emotions = {'Positive': 0, 'Negative': 1, 'Neutral': 2 }
kaggle.replace({'Labels':rep_emotions},inplace=True)

import sys
sys.path.append("..") 
import demo_functions
SAMPLE_RATE = 16000


data = {
        "subject": [],
        "labels": [],
        "dt": []
    }


for i, row in kaggle.iterrows():
    data['subject'].append(row['file'])
    data['labels'].append(row['emotion'])
    try:
        print('opening', row['file'])
        embeddings = demo_functions.run_DeepTalk_demo(ref_audio_path=path+row['file'])
    except:
        print('bad embedding')
        embeddings = None
        data['labels'][i] = 'discard'

    data["dt"].append(np.asarray(embeddings))
    if i%500==0:
        print(i)
    

data_df = pd.DataFrame(data)
data_df.to_pickle('./working_data/kaggle_extracted_embeddings_test.pk')
