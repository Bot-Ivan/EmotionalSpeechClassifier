from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

Crema = "C:\\Users\\ivanf\\OneDrive\\Desktop\\EmotionalSpeechClassifier\\AudioWAV" #audio files path

crema_directory_list = os.listdir(Crema) #listdir returns a LIST of all the files in the specified diretory

file_emotions = []
file_paths = []

for file in crema_directory_list:
    file_paths.append(Crema + file)

    part = file.split('_')

    if part[2] == 'SAD':
        file_emotions.append('sad')
    elif part[2] == 'ANG':
        file_emotions.append('angry')
    elif part[2] == 'DIS':
        file_emotions.append('disgust')
    elif part[2] == 'FEA':
        file_emotions.append('disgust')
    elif part[2] == 'HAP':
        file_emotions.append('happy')
    elif part[2] =='NEU':
        file_emotions.append('neutral')
    else:
        file_emotions.append('Unknown')

emotion_df = pd.DataFrame(file_emotions, columns=['Emotions'])

path_df = pd.DataFrame(file_paths, columns = ['Path'])

Crema_df = pd.concat([emotion_df, path_df], axis = 1)

print(Crema_df.head())

data_path = Crema_df
data_path.to_csv("data_path.csv", index = False)

Crema_plt = Crema_df

le = LabelEncoder()
Crema_plt['Emotions_Encoded'] = le.fit_transform(Crema_plt['Emotions'])






plt.title('Count of Emotions', size = 16)
sns.countplot(x='Emotions_Encoded', data=Crema_plt)
plt.ylabel('Count', size = 12)
plt.xlabel('Emotions', size = 12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()



