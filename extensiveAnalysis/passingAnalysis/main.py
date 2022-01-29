import pandas as pd
import numpy as np
from copy import deepcopy
import glob
import sys

files = glob.glob('../data/*.csv')
eventsDataLaLiga2019 = pd.read_csv(files[1], low_memory=False)
seasonMetaDataLaLiga2019 = pd.read_csv(files[2], low_memory=False)

## type.id 30 --> Event [pass]
eventPassdf = deepcopy(eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 30])

passAdditionalCols = ['match_id', 'id',
                      'timestamp', 'minute', 'second',
                      'type.id', 'type.name',
                      'possession_team.id', 'possession_team.name',
                      'play_pattern.id', 'play_pattern.name',
                      'team.id', 'team.name', 'location',
                      'player.id', 'player.name',
                      'position.id', 'position.name',
                      'under_pressure',
                      'started', 'minsPlayed']

pass_cols = eventPassdf.columns[eventPassdf.columns.str.startswith("pass")].tolist()
pass_cols = passAdditionalCols + pass_cols

## slice the required columns

eventPassdf = eventPassdf[pass_cols]

eventPassdf.shape
eventPassdf.nunique(axis=0)

"""## Separate Categorical and Numerical Columns"""

eventPassdf.select_dtypes(include="category").columns
eventPassdf.select_dtypes(include="object").columns

catPassCols = eventPassdf.select_dtypes(include="object").columns
## remove unwanted columns from categorical df

unwantedCols = ["player.name", "pass.recipient.name",
                "pass.end_location", "location",
                "pass.assisted_shot_id", "id", "timestamp",
                "position.name", "possession_team.name",
                "team.name", "type.name"]

catPassCols = catPassCols.symmetric_difference(unwantedCols)
catPassCols.__len__()


"""## *Memory optimization __ loading or performing operations on categorical is faster than object ##"""

eventPassdf["pass.height.name"].value_counts()

# Commented out IPython magic to ensure Python compatibility.
# %timeit eventPassdf["pass.height.name"].value_counts()
eventPassdf["pass.height.name"] = eventPassdf["pass.height.name"].astype("category")
# Commented out IPython magic to ensure Python compatibility.
# %timeit eventPassdf["pass.height.name"].value_counts()

eventPassdf[catPassCols] = eventPassdf[catPassCols].astype("category")

## Numerical columns, most are id's thus take only what is required
eventPassdf.select_dtypes(exclude=["object", "category"]).columns
numPassCols = ['pass.length', 'pass.angle', 'minsPlayed']



"""## Numerical Data Analysis"""
## a lot of columns have nan which need to be converted into False and added as a category ##


eventPassdf.info()
eventPassdf[numPassCols].describe().T.round(1)
eventPassdf.isnull().sum()

booleanCols = []
for col in eventPassdf[catPassCols]:
    print(col, "\n", eventPassdf[col].unique(),' ', list(eventPassdf[col].unique()) , "\n")
    if  len(eventPassdf[col].unique()) == 2 and np.nan in list(eventPassdf[col].unique()):
        print(col)
        booleanCols.append(col)
booleanCols.remove('started')

eventPassdf[["pass.outcome.id", "pass.outcome.name"]].drop_duplicates()

## pass.outcome -> nan/null meaning the pass was complete ##
eventPassdf["pass.outcome.id"] = np.where(eventPassdf["pass.outcome.id"].isnull(), 1,
                                          eventPassdf["pass.outcome.id"])
eventPassdf["pass.outcome.name"] = np.where(eventPassdf["pass.outcome.name"].isnull(), "Complete",
                                            eventPassdf["pass.outcome.name"])

eventPassdf[["pass.outcome.id", "pass.outcome.name"]].drop_duplicates()

## need to first add False to the categories before using fillna ##
for col in booleanCols:
    try:
        eventPassdf[col] = eventPassdf[col].cat.add_categories(False)
    except ValueError:
        print(sys.exc_info())
    eventPassdf[col] = eventPassdf[col].fillna(False)

for col in catPassCols:
    print(col, "\n", round(eventPassdf[col].value_counts(normalize=True)*100, 3), "\n")



"""## Visual Data Analysis"""

import mplsoccer
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages


## basic histograms plotted ##
eventPassdf[numPassCols].hist(bins=10, figsize=(16, 8))

from IPython.display import Image
Image("./Pass.Angle.PNG")

##
'''
angle = 0 -> opposition goal
angle = -90(1.58 rad) to 90 -> forward pass *can be defined as per use case
angle = 90 to 180(3.14 rad) or -90 to -180 -> backward passes
'''
##

eventPassdf["play_pattern.name"].value_counts().reset_index()


## interactive histograms for various stats, right foot passes are high, passes under pressure are low,
## most passes are complete, ground passes are high, etc

i = j = 1
# Create an empty subplot:
fig = make_subplots(rows=5, cols=5,
                    shared_xaxes=False,
                    vertical_spacing=0.1,
                    subplot_titles=catPassCols)
# Add bar plot for goals conceded in all subplots:
for col in catPassCols:
    plot_df = eventPassdf[col].value_counts().reset_index()
    fig.add_trace(go.Bar(x=plot_df["index"],
                         y=plot_df[col],
                         text=plot_df[col],
                         name=col),
                  row=i, col=j)
    j += 1
    if j > 5:
        j = 1
        i += 1
fig.update_traces(textposition='inside', textfont_size=10)
fig.update_layout(height=1600)

fig.show()


"""## Team Data Analysis"""

## match_id - nunique to get the number of matches played by that team
teamWisedf = eventPassdf.groupby("team.id").agg({"team.name": "first", "match_id": "nunique", "type.id": "count"})
teamWisedf.columns = ["teamName", "nMatches", "nPasses"]

## since inconsistent num of matches played, perMatchPasses is calculated
teamWisedf["passesPerMatch"] = teamWisedf["nPasses"].divide(teamWisedf["nMatches"])

## highlights the max value in column
teamWisedf.style.highlight_max().set_precision(1)

## creates a bar chart within df, add subset to single out columns
teamWisedf.style.bar().set_precision(1)
teamWisedf.style.bar(subset=["passesPerMatch"]).set_precision(1)

def color_avg_red(val):
    """
    computes every column of df
    if dtype object no color is returned
    else based on current value in respect to average value
    green/red marking is given
    """
    if val.dtype == "object":
        return [""]*len(val)
    valMean = val.mean()
    colors = ['color: green' if (v > valMean) else 'color: red' for v in val]
    return colors

teamWisedf.style.apply(color_avg_red).set_precision(1)


"""### Top Passing Players"""

playerWisedf = eventPassdf.groupby("player.id").agg({"player.name": "first", "team.name": "first",
                                                     "match_id": "nunique", "type.id": "count"})
playerWisedf["passesPerMatch"] = playerWisedf["type.id"].divide(playerWisedf["match_id"])

## gives the players whose passing is above average 'green' and below average 'red'
playerWisedf.style.apply(color_avg_red).set_precision(1)

avgPassesPerMatch = playerWisedf["passesPerMatch"].mean()

playerWisedf[playerWisedf["passesPerMatch"] > avgPassesPerMatch].style.apply(color_avg_red).set_precision(1)
playerWisedf[playerWisedf["passesPerMatch"] > avgPassesPerMatch].style.highlight_max()
playerWisedf[playerWisedf["passesPerMatch"] > avgPassesPerMatch].style.bar().set_precision(1)

## since taking the num passes per match is not a great indicator
## as a player could have played n number of minutes, we can use the minsPlayed engineered
## feature

"""### Top Passing Players - Per90"""

## since event data, one match can have n passing events thus dropping based on player and match id
## followed by grouping on player id and agg on minsPlayed

playerMatchMinsdf = eventPassdf.drop_duplicates(subset=["player.id", "match_id"]).groupby(["player.id"])\
.agg({"minsPlayed": "sum"})

## num matches played by each player
playerWisedf = eventPassdf.groupby("player.id").agg({"player.name": "first", "team.name": "first",
                                                     "match_id": "nunique", "type.id": "count"})
## joining the 2 dataframes horizontally since indexes are same 
playerWisedf = pd.concat([playerWisedf, playerMatchMinsdf], axis=1)

playerWisedf["passesPerMatch"] = playerWisedf["type.id"].divide(playerWisedf["match_id"])
playerWisedf["passesPer90"] = playerWisedf["type.id"].divide(playerWisedf["minsPlayed"])*90

avgPassesPer90 = playerWisedf["passesPer90"].mean()

## here its important to get per 90 stats on players as that paints a bigger picture 
## instead of passes per match
playerWisedf[playerWisedf["passesPer90"] > avgPassesPer90].style.bar(subset=["passesPerMatch",
                                                                             "passesPer90"]).set_precision(1)