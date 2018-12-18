
# coding: utf-8

# In[7]:

import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras import layers
lyricsdataframe=pd.read_csv('./finallyrics1.csv',error_bad_lines=False)
X=lyricsdataframe['Lyrics']
Y=lyricsdataframe['Mood']
laen=LabelEncoder()
y=laen.fit_transform(Y)
from keras.utils import np_utils
y=np_utils.to_categorical(y)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2)
def tokkk(x):
    max_words1 = 550
    max_len1 = 600
    lyricstok = Tokenizer(num_words=max_words1)
    lyricstok.fit_on_texts(x)
    seque = lyricstok.texts_to_sequences(x)
    sequen_matrix = sequence.pad_sequences(seque,maxlen=max_len1)
    return sequen_matrix,max_len1,max_words1
sequences_matrix,max_len,max_words = tokkk(X_train)
inp = Input(name='inputs',shape=[max_len])
i = Embedding(max_words,50,input_length=max_len)(inp)
i = LSTM(32)(i)
i = Dense(256, name='FC1')(i)
i = Activation('relu')(i)
i = Dropout(0.2)(i)
i = Dense(4,name='out_layer')(i)
out = Activation('softmax')(i)
model = Model(inputs=inp,outputs=out)
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history=model.fit(sequences_matrix,Y_train,batch_size=128,epochs=40,validation_split=0.2)
lyricsdataframe1=pd.read_csv('./new/songdata.csv',error_bad_lines=False)
XX=lyricsdataframe1['text']
xx,max_len,max_words=tokkk(XX)
mm=model.predict(xx)
mmm=[]
for i in range(len(mm)):
    top=max(mm[i])
    for j in range(len(mm[0])):
        if(mm[i][j]==top):
            mmm.append(j)
with open('./56k2.csv',mode='a') as k:
    k_writer = csv.writer(k, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    k_writer.writerow(['serial','Artist','Title','Lyrics','mood'])
    for i in range(len(mmm)):
        k_writer.writerow([i,lyricsdataframe1['artist'][i],lyricsdataframe1['song'][i],lyricsdataframe1['text'][i],mmm[i]])

