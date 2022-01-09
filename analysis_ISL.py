# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:44:29 2021

@author: Agnij
"""

# import io
import pandas as pd
import numpy as np
from copy import deepcopy

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

isl_df = pd.read_csv('./datasets/isl_player_final.csv')
stringColumns = list(isl_df.loc[:, isl_df.dtypes == object].columns)
columnNames = sorted(list(isl_df.columns))

isl_df_1 = isl_df.drop(columns = stringColumns)
isl_df_1 = isl_df_1.apply(pd.to_numeric, errors='ignore')

# isl_df_season6
# isl_df_season7

## Contains data for 2 seasons ISL_6 an ISL_7 || Should seperate out the seasons before Comparing
isl_df_1_statistics = deepcopy(isl_df_1.describe().round(1))
isl_df[isl_df_1['minutes_played'] == 2100]['is_started']

####
## Indian Forwards Analysis || comparison with Sunil Chettri
####

indian_forwards = deepcopy(isl_df[isl_df["country_id"] ==1 & (isl_df["position_id"] ==2 )])
indian_forwards.reset_index(drop= True, inplace = True)

indian_forwards[indian_forwards['name'].str.contains("Sunil")]['team_name']

forwards_minutes = indian_forwards.groupby(["id", "name"])["minutes_played"].sum().reset_index()

## Check does not plot on spyder3
## px.bar(x= "name", y="minutes_played", text="minutes_played", data_frame=forwards_minutes)

g = sns.barplot(x=forwards_minutes['name'].to_list(), y=forwards_minutes['minutes_played'].to_list())
g.set_xticklabels(labels = forwards_minutes['name'].to_list(), rotation=90)
plt.show()


## per 90 stats
## indian_forwards['goalsPer90'] = indian_forwards['events.goals'].divide(indian_forwards['minutes_played']).multiply(90)
## indian_forwards.drop(columns = ['goalsPer90'], inplace= True)
per90Cols = ["events.goals", "events.assists", "events.shots", "events.shots_on_target", "events.chances_created",
             "events.fouls_suffered", "touches.total", "touches.aerial_duel.won", "touches.ground_duel.won"]

for col in per90Cols:
    indian_forwards["Per90"+col] = indian_forwards[col].divide(indian_forwards['minutes_played']).multiply(90)
    
cols_for_radar = ["Per90" + i  for i in per90Cols]
indian_forwards.loc[(indian_forwards['id'] == 19150), cols_for_radar].sum()


for i, row in indian_forwards.iterrows():
    print(row['name'])
    fig = px.line_polar(indian_forwards, r=indian_forwards.loc[(indian_forwards["id"] == row["id"]), cols_for_radar].sum(),
                                                                theta=cols_for_radar, line_close=True,
                                                                title = row["name"])
    fig.update_traces(fill = 'toself')
    # fig.show()
    px.iplot(fig)
    break


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
indian_forwards[cols_for_radar] = scaler.fit_transform(indian_forwards[cols_for_radar])


## Multiple Trace Radar chart
    
ind_fwd_id_names = indian_forwards.drop_duplicates(subset=["id"])[["id", "name"]]
ind_fwd_id_names.head()

## Some sort of scaling Eh!
isl_max = indian_forwards[cols_for_radar].max().max()

import plotly.graph_objects as go

for i, row in ind_fwd_id_names.iterrows():
    if row['id'] == 19150:
        continue
    
    print(row['name'])
    ## plotly graph initiation
    fig = go.Figure()
    
    ## Adding Radar plot for different figures
    ## Sunil's plot
    fig.add_trace(go.Scatterpolar(
            r = indian_forwards.loc[(indian_forwards["id"] == 15190), cols_for_radar].sum(),
            theta = cols_for_radar, fill = 'toself', name = 'Sunil Chhetri'
            ))
    ## 
    ## In Comparison player 
    fig.add_trace(
            r = indian_forwards.loc[(indian_forwards["id"] == row["id"]), cols_for_radar].sum(),
            theta = cols_for_radar, fill = 'toself', name = row["name"]
            )
    
    ## Additional properties
    fig.update_layout(
            title = "Sunil Chhetri vs " + row["name"],
            polar=dict(radialaxis = dict(
                    visible = True,
                    range = [0, isl_max]
                    )),
            showlegend = True
            )
    fig.show()
    
## Manvir Singh || Liston Colaco || Leon Augustin
indian_forwards[indian_forwards['name'].str.contains("Pandita")]['minutes_played']

komalThatal = indian_forwards[indian_forwards['name'].str.contains("Komal")]    