# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:58:08 2021

@author: Agnij
"""

import json
import requests
import pandas as pd

## sofascore.com 
'''
try: getting data from the webpage HTML
        right click -> inspect
        right click -> view pagesource

else:
        right click -> inspect
        right click -> Network -> XHR -> will contain all APIs
        right click [relevant] to obtain url
'''

scrape_url = "https://api.sofascore.com/api/v1/unique-tournament/1900/season/34713/standings/total"
r = requests.get(scrape_url)
json_object = json.loads(r.content)

json_object['standings'][0].__len__()
json_object['standings'][0].keys()

df = pd.json_normalize(json_object['standings'][0]['rows'])

'''
import iso4217parse
text = iso4217parse.by_symbol_match('GBP10,000,000 ')
print(str(text[0][3][1])+ ' <--> ' +str(text[0][2].title()))
'''
