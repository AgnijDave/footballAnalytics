# -*- coding: utf-8 -*-
"""
Created on Fri May  7 08:09:23 2021

@author: Agnij
"""

import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
from copy import deepcopy

'''

BeautifulSoup is used Since the webpage
contains all data in string/object format.
    -This requires the steps involved to encode
     and decode using the json lib.

if an API is found, it can directly be scraped
using requests lib. since it will already in
JSON (dictionary) format

'''

## make a list of urls to scrape ## urls = []
scrape_url = "https://understat.com/league/EPL"
page_connect = urlopen(scrape_url)

page_html = BeautifulSoup(page_connect, "html.parser")

## use argument -> attrs <- to specify code blocks to be extracted
html_script = page_html.find_all(name = "script")
html_script_len = page_html.find_all(name = "script").__len__()

json_raw_string = html_script[3].text
start_index = json_raw_string.index("('")+2
end_index = json_raw_string.index("')")

json_data = json_raw_string[start_index:end_index]
## string format (JSON)
json_data_dictionary = json_data.encode("utf-8").decode("unicode_escape")
## loading the string format (JSON) as a json object :: list of dictionaries  
## json_data_dictionary = json.loads(json_data_dictionary)

final_json = pd.json_normalize(json.loads(json_data_dictionary))
final_json.assists.unique()

assists = final_json.groupby(['player_name'])['assists'].unique()
## assists['Harry Kane']
assists_df = assists.to_frame()

for index, row in assists_df.iterrows():
    print(index,'  >--< ', row['assists'][0])
    
final_json.describe().round(1)



## ALL THE DATA will of the ||object|| datatype since a JSON is used to load it into the DataFrame
## final_json.loc[:, final_json.dtypes == object]

final_json_ = final_json.drop(columns = ['player_name', 'position', 'team_title'])
final_json_ = final_json_.apply(pd.to_numeric, errors = 'ignore')

final_json_statistics = deepcopy(final_json_.describe().round(1))

'''
df = pd.DataFrame({'day':[1,2,12,243,5], 'month':[3242,1231,56,7
                   ,43], 'year':[1998, 1203, 4990, 1203,1000]})
df.describe()
'''

## Get The Details of a Single Player
final_json[final_json['player_name'].str.contains("Alioski")]
final_json[final_json['goals'] == '21']
