from pandas import read_parquet
import numpy as np
import pandas as pd
import os
import re

# unlike Kaggle and MELD, there is no cvs that we can use as "labels"
# Specifcally, each digit in the title of the wav file represents the emotions
'''
Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong).  There is no strong intensity for the 'neutral' emotion.
Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
'''

wavs = "Emotion Recognition/Data/RAVDESS/Audio_Speech_Actors_01-24"

list = os.listdir(wavs)

file_emotion = []
file_path = []
file_label = [] #by this we mean pos, neu, or neg

for file in list:
    digits = re.findall(r'\d+', file) #each digit as mentioen erlier means different things
    file_path.append(wavs + file)
    
    if digits[2] == '04':
        file_emotion.append('sad')
        file_label.append('neg')
    elif digits[2] == '05':
        file_emotion.append('ang')
        file_label.append('neg')
    elif digits[2] == '07':
        file_emotion.append('other')
        file_label.append('neg')
    elif digits[2] == '06':
        file_emotion.append('other')
        file_label.append('neg')
    elif digits[2] == '03':
        file_emotion.append('happy')
        file_label.append('pos')
    elif digits[2] == '01':
        file_emotion.append('neu')
        file_label.append('neu')
    elif digits[2] == '02': #calm and neu put in same
        file_emotion.append('neu')
        file_label.append('neu')
    elif digits[2] == '08': 
        file_emotion.append('other')
        file_label.append('neg')
    else:
        file_emotion.append('other')
        file_label.append('other')
        

emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
labels_df = pd.DataFrame(file_label, columns=['Labels'])
path_df = pd.DataFrame(file_path, columns=['Path'])

RAVDESS_df = pd.concat([labels_df, emotion_df, path_df], axis=1)

RAVDESS_df.to_parquet('./working_data/RAVDESS.pq')
print('Successfully saved the DF to a parquet in ./working_data')
RAVDESS_df = read_parquet('./working_data/RAVDESS.pq')

rep_emotions = {'pos': 0, 'neg': 1, 'neu': 2, 'other':10}
labels = {'hap':1,'sad':2,'neu':3,'ang':4, 'other':10}
RAVDESS_df.replace({'Emotions':labels},inplace=True)
RAVDESS_df.replace({'Lables':rep_emotions},inplace=True)

import sys
sys.path.append("..") 
import deep_functions
SAMPLE_RATE = 16000

data = {
        "subject": [],
        "labels": [],
        "dt": []
    }

for i in range(7442):
    print(i)
    data['subject'].append(RAVDESS_df.iloc[i,2])
    data['labels'].append(RAVDESS_df.iloc[i,0])

    try:
        print('opening', RAVDESS_df.iloc[i, 0])
        #embeddings = deep_functions.run_DeepTalk_demo(ref_audio_path=( RAVDESS_df.iloc[i, 0]))
    except:
        print('bad embedding')
        embeddings = None
        data['labels'][i] = 'discard'
    data["dt"].append(np.asarray(embeddings))
    if i%500==0:
        print(i)


data_df = pd.DataFrame(data)
data_df.to_pickle('./working_data/RAVDESS_extracted_embeddings.pk')