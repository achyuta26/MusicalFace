#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:00:56 2018

@author: achyutajha
"""

import cv2
import numpy as np
from keras.models import load_model
import argparse
import pandas as pd
import time
import os





ap = argparse.ArgumentParser()
ap.add_argument("-c", "--choice", required=True, type=int,
help="choice of image : 0-webcam, any other integer- saved image",)
ap.add_argument("-m", "--modelpath", required=True,
help="path to saved model",)

args = vars(ap.parse_args())
mypath=args['choice']
path_to_model=args['modelpath']





face_expression_map = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
emotion_map = {0:'angry',1:'happy',2:'relaxed',3:'sad' }

def img_to_matrix(imagePath):
    image=cv2.imread(imagePath)
    image=cv2.resize(image, (48,48))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def emotion_label_to_song_label(emotion):

    song_emotion=None
    if(emotion==0 or emotion==1):
        song_emotion=0
    elif(emotion==2 or emotion==4):
        song_emotion=3
    elif(emotion==3):
        song_emotion=1
    else:
        song_emotion=2       
    return song_emotion







##################### Take User Choice of input#################

if mypath ==0:
    from webcam_util import capture_image
    image_path = capture_image()
    print image_path
else:
    print "Enter path to image:"
    image_path = str(input())



start_time = time.time()

##################### Preprocess image before prediction ################

if os.path.exists(image_path):
    image = img_to_matrix(image_path)
    image=image.astype('float32')
    image/=255
    image = image.reshape(48,48,1)
    test_image = np.expand_dims(image, axis = 0)


    ################# Predict input from saved model############################
    model = load_model(path_to_model)
    result = model.predict(test_image)

    resultList = list(result[0])
    predicted_face_emotion = resultList.index(max(resultList))
    print "\n\n\n\n\n"
    print "######################################################"
    print 'Predicted Emotion:',face_expression_map[predicted_face_emotion]
    print 'Result Set:\t',resultList
    print 'Top 3 predictions:',sorted(resultList,reverse=True)[:3]
    
    
    
    ############ Narrow down from 6 categories of face emotion to 4 available for face ############
    
    song_emotion = emotion_label_to_song_label(predicted_face_emotion)
    
    
    labelled_song = pd.read_csv('/Users/achyutajha/Documents/PSU Study Mat/Fall-II/Deep Learning/Project/Data/MoodyLyrics/56k2.csv')
    mood_songs = labelled_song[labelled_song.mood == song_emotion]
    
    #Genenrate sample of random 10 songs from the selected mood songs
    playlist_DF = mood_songs.sample(10,random_state=np.random.RandomState())
    
    
    ################ Retrieve Spotify URL for 10 songs based on mood ################    
    from SpotifyUtil import SpotifyUtil
    #to-do : paste your own client_id and client_secret from Spotify developer dashboard
    client_id=''
    client_secret=''
    spUtil = SpotifyUtil(client_id,client_secret)
    
    list_of_track_urls = list()
    for _,record in playlist_DF.iterrows():
        list_of_track_urls.append(spUtil.get_url_of_track(record.Artist,record.Title))
    
    list_artist_title_urls = zip(playlist_DF.Artist,playlist_DF.Title,list_of_track_urls)
    artist_title_urls_df= pd.DataFrame(list_artist_title_urls, columns=['Artist','Title','URL'])
    print "######################################################"
    print "\n\n\n\nHey %s face these are the list of your songs:\n\n"%emotion_map[song_emotion]
    print artist_title_urls_df
    print "\n\nExecution time:%s seconds"%(time.time()-start_time)
    
    ################## Launching songs on Web browser ##################################
    
    print "\n\n\n\n\n"
    print "Enjoy playlist on default browser!"
    print "\n\n\n\n\n"
    import webbrowser    
    webbrowser.open_new_tab(artist_title_urls_df.URL[0])
    
else:
    print "Invalid Path:",image_path
