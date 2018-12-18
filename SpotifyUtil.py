#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 09:41:30 2018

@author: achyutajha
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyUtil(object):
  
    def __init__(self,client_id,client_secret):
        self.client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
    
    def get_url_of_track(self,artist,name):
        combinedResult = self.sp.search(q='artist:{} track:{}'.format(str(artist), str(name)))
        url_of_track = None
        try:
            url_of_track = str(combinedResult['tracks']['items'][0]['external_urls']['spotify'])
        except Exception as e:
            url_of_track = 'https://open.spotify.com/track/5wDPLdzR7eivEl4Dkd7tST' #default track in case of Key error in resultset
        
        return url_of_track 
    
