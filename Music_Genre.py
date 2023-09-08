#!/usr/bin/env python
# coding: utf-8

# ## Genre Recommendation 

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('data.csv',encoding='latin1')


# In[3]:


df = df.drop(['beats'], axis=1)


# In[4]:


df.head()


# In[5]:


df['class_name'].unique()


# In[6]:


df['class_name'] = df['class_name'].astype('category')
df['class_label'] = df['class_name'].cat.codes


# In[7]:


lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))   
lookup_genre_name


# In[8]:


df['class_name'].unique()


# In[1]:


cols = list(df.columns)
cols.remove('label')
cols.remove('class_label')
cols.remove('class_name')
#df[cols]


# In[10]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
X = df.iloc[:,1:28]
y = df['class_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
# we must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test) 


# In[12]:


from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'notebook')
clf = RandomForestClassifier(random_state=0, n_jobs=-1).fit(X_train_scaled, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
names = [X.columns.values[i] for i in indices]
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), names, rotation=90)
plt.show()


# ## SVM with all 27 features 

# In[13]:


get_ipython().system('pip install librosa')


# In[14]:


def getmetadata(filename):
    import librosa
    import numpy as np


    y, sr = librosa.load(filename)
    #fetching tempo

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    #fetching beats

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

    #chroma_stft

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    #rmse

    rmse = librosa.feature.rms(y=y)

    #fetching spectral centroid

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    #spectral bandwidth

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    #fetching spectral rolloff

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    #zero crossing rate

    zero_crossing = librosa.feature.zero_crossing_rate(y)

    #mfcc

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    #metadata dictionary

    metadata_dict = {'tempo':tempo,'chroma_stft':np.mean(chroma_stft),'rmse':np.mean(rmse),
                     'spectral_centroid':np.mean(spec_centroid),'spectral_bandwidth':np.mean(spec_bw), 
                     'rolloff':np.mean(spec_rolloff), 'zero_crossing_rates':np.mean(zero_crossing)}

    for i in range(1,21):
        metadata_dict.update({'mfcc'+str(i):np.mean(mfcc[i-1])})

    return list(metadata_dict.values())


# In[15]:


a = getmetadata("sample1.wav")


# In[16]:


from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10, gamma=0.9, probability=False).fit(X_train_scaled, y_train)
clf.score(X_test_scaled, y_test)


# In[17]:


d1 =np.array(a)
data1 = scaler.transform([d1])
genre_prediction = clf.predict(data1)
genre_name = lookup_genre_name[genre_prediction[0]]


# In[18]:


print(genre_name)


# ## Song Recommendation 

# In[19]:


import pandas as pd

def song_recommendation(genre_name):
    # read the CSV file into a DataFrame
    songs_df = pd.read_csv('musicc.csv', encoding='Windows-1252')

    # filter the DataFrame to only include rows with the predicted genre
    filtered_df = songs_df[songs_df['genre'] == genre_name]

    # get a list of 5 random song names with the predicted genre
    song_names = filtered_df['song_name'].sample(n=5).tolist()

    # return the list of song names
    return song_names


# In[20]:


# call the function with the predicted genre name
recommended_songs = song_recommendation(genre_name)

# print the recommended songs
print(recommended_songs)

