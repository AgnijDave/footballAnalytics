# -*- coding: utf-8 -*-
"""
Created on Sun May 16 18:29:42 2021
@author: Agnij
Messi vs Ronaldo
"""

import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from sklearn.preprocessing import MinMaxScaler

## Get the Relevant Data
try:
    season_wise_meta_df = pd.read_csv('./datasets/seasonWiseMetaData.csv')
    shots_df = pd.read_csv('./datasets/shotsData.csv') 
    print('Data -> Found Locally\n')
except:    
    ## CR7 Season Wise Data + Shots Data
    ## 1
    try:
        cr7_season_wise_meta_df = pd.read_csv('./datasets/cr7SeasonData.csv')
        print('Success -> CR7-Season-Data\n')
    except:
        print('Downloading -> CR7-Season-Data\n')
        cr7 = "https://understat.com/player/2371"
        page_connect = urlopen(cr7)
        page_html = BeautifulSoup(page_connect, "html.parser")
        
        json_raw_string = page_html.findAll(name="script")[1].text
        start_ind, stop_ind = json_raw_string.index("\\"), json_raw_string.index("')")
        
        json_data = json_raw_string[start_ind:stop_ind]
        json_data = json_data.encode("utf-8").decode("unicode_escape")
        
        cr7_season_wise_meta_df = pd.json_normalize(json.loads(json_data)["season"])
        cr7_season_wise_meta_df.insert(0, "Player", "CR7")
        cr7_season_wise_meta_df.to_csv('./datasets/cr7SeasonData.csv')
    
    
    ## 2
    try:
        cr7_shots_df = pd.read_csv('./datasets/cr7ShotsData.csv')
        print('Success -> CR7-Shots-Data\n')
    except:
        print('Downloading -> CR7-Shots-Data\n')
        json_raw_string = page_html.findAll(name="script")[3].text
        start_ind, stop_ind = json_raw_string.index("\\"), json_raw_string.index("')")
        
        json_data = json_raw_string[start_ind:stop_ind]
        json_data = json_data.encode("utf-8").decode("unicode_escape")
        
        cr7_shots_df = pd.json_normalize(json.loads(json_data))
        cr7_shots_df.insert(0, "Player", "CR7")
    
        cr7_shots_df.to_csv('./datasets/cr7ShotsData.csv')    
    
    
    ## Messi Season Wise Data + Shots Data
    ## 1
    try:
        messi_season_wise_meta_df = pd.read_csv('./datasets/messiSeasonData.csv')
        print('Success -> Messi-Season-Data\n')
        
    except:
        print('Downloading -> Messi-Season-Data\n')
        messi = "https://understat.com/player/2097"
        page_connect = urlopen(messi)
        page_html = BeautifulSoup(page_connect, "html.parser")
        
        json_raw_string = page_html.findAll(name = "script")[1].text
        start_ind, stop_ind = json_raw_string.index("\\"), json_raw_string.index("')")
        
        json_data = json_raw_string[start_ind:stop_ind]
        json_data = json_data.encode("utf-8").decode("unicode_escape")
        
        messi_season_wise_meta_df = pd.json_normalize(json.loads(json_data)["season"])
        messi_season_wise_meta_df.insert(0, "Player", "Messi")
        messi_season_wise_meta_df.to_csv('./datasets/messiSeasonData.csv')
        
    ## 2
    try:
        messi_shots_df = pd.read_csv('./datasets/messiShotsData.csv')
        print('Success -> messi-Shots-Data\n')
    except:
        print('Downloading -> Messi-Shots-Data\n')
        json_raw_string = page_html.findAll(name="script")[3].text
        start_ind, stop_ind = json_raw_string.index("\\"), json_raw_string.index("')")
        
        json_data = json_raw_string[start_ind:stop_ind]
        json_data = json_data.encode("utf-8").decode("unicode_escape")
        
        messi_shots_df = pd.json_normalize(json.loads(json_data))
        messi_shots_df.insert(0, "Player", "Messi")
        
        messi_shots_df.to_csv('./datasets/messiShotsData.csv')
        
    ## Combine the Data
    season_wise_meta_df = cr7_season_wise_meta_df.append(messi_season_wise_meta_df)
    season_wise_meta_df.to_csv('./datasets/seasonWiseMetaData.csv')

    print('Saving -> Combined Data\n')
    shots_df = cr7_shots_df.append(messi_shots_df)
    shots_df.to_csv('./datasets/shotsData.csv')

    
        
        
    
    
    