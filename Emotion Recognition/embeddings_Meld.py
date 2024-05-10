import numpy as np
import csv
import re
import os
import pandas as pd

csv_file = "Emotion Recognition/Data/MELD/dev_sent_emo.csv"
path_mp4s = "Emotion Recognition/Data/MELD/dev_splits_complete"

list = os.listdir(path_mp4s)

file_emotion = []
file_path = []
file_label = [] #by this we mean pos, neu, or neg
csv_parse = []

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        line_str = ','.join(line)
        first_quote_index = line_str.index('"')
        second_quote_index = line_str.index('"', first_quote_index + 1)

        substring = line_str[second_quote_index + 1:]

        words = substring.split(',')

        words = [word.strip() for word in words]
        csv_parse.append(words)

for file in list:
    digits = re.findall(r'\d+', file)
    for file_digit in csv_parse:
        if file_digit[3] == digits[0] and file_digit[4] == digits[1]: #takes emotion and label based of finding file in csv
            file_emotion.append[file_digit[1]]
            file_label.append[file_digit[2]]
            file_path.append(path_mp4s + file)
    

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
labels_df = pd.DataFrame(file_label, columns=['Labels'])
path_df = pd.DataFrame(file_path, columns=['Path'])

data_set = pd.concat([labels_df, emotion_df, path_df], axis=1)


data_set.to_parquet('./working_data/MELD.pq')
print('Successfully saved the DFs to a parquet in ./working_data')


MELD = read_parquet('./working_data/MELD.pq')

rep_emotions = {'positive': 0, 'negative': 1, 'neutral': 2 }
labels = {'joy':1,'sadness':2,'neutral':3,'anger':4, 'fear': 10, 'suprise': 10, 'disgust': 10} #labels are different in this datset hence the change
MELD.replace({'Emotions':labels},inplace=True)
MELD.replace({'Labels':rep_emotions},inplace=True)

import sys
sys.path.append("..") 
import demo_functions
SAMPLE_RATE = 22050

data = {
        "subject": [],
        "labels": [],
        "dt": []
    }

for i in range(len(MELD)):
    data['subject'].append(MELD.iloc[i,2])
    data['labels'].append(MELD.iloc[i,1])
    
    utterances_path='Emotion Recognition/Data/MELD/dev_splits_complete'
    try:
        print('opening', utterances_path + MELD.iloc[i, 0])
        embeddings = demo_functions.run_DeepTalk_demo(ref_audio_path=(utterances_path +MELD.iloc[i, 0]))
    except:
        print('bad embedding')
        embeddings = None
        data['labels'][i] = 'discard'




    data["dt"].append(np.asarray(embeddings))
    if i%500==0:
        print(i)


data_df = pd.DataFrame(data)
data_df.to_pickle('./working_data/MELD_extracted_embeddings.pk')