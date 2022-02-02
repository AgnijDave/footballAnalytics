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


"""# Generating and Analysing Pass Maps"""

## getting the nodes and edges for plotting using mplsoccer ##
## expand = True, gives a string instead of a list ##

eventPassdf["startX"] = eventPassdf["location"].str.split(", ", expand=True)[0].str[1:].apply(pd.to_numeric)
eventPassdf["startY"] = eventPassdf["location"].str.split(", ", expand=True)[1].str[:-1 ].apply(pd.to_numeric)

eventPassdf["endX"] = eventPassdf["pass.end_location"].str.split(", ", expand=True)[0].str[1:].apply(pd.to_numeric)
eventPassdf["endY"] = eventPassdf["pass.end_location"].str.split(", ", expand=True)[1].str[:-1].apply(pd.to_numeric)

'''
## based on matplotlib library ##

pitch = mplsoccer.Pitch(pitch_color='#101010', line_color='white')
fig, ax = pitch.draw(figsize=(14, 10)) ## width and height

## generate arrows function, startx, starty, endx, endy
arrows = pitch.arrows(eventPassdf["startX"], eventPassdf["startY"],
                      eventPassdf["endX"], eventPassdf["endY"],
                      ax=ax,
                      width=.1,
                      color="green")

'''
## get analysis on a particular match ## Atletico Madrid vs Barcelona [303696] ##
eventPassdf[["match_id", "team.name"]].drop_duplicates()

matchEventdf = eventPassdf[eventPassdf["match_id"] == 303696]
pitch = mplsoccer.Pitch(pitch_color='#101010', line_color='white')
fig, ax = pitch.draw(figsize=(14, 10))

arrows = pitch.arrows(matchEventdf["startX"], matchEventdf["startY"],
                      matchEventdf["endX"], matchEventdf["endY"],
                      ax=ax,
                      width=1,
                      color="green")

## analysis keeping segregating team passing into differet colors ##
## Real Madrid vs Barcelona [303470] ##

team_ids = eventPassdf[eventPassdf["match_id"] == 303470]["possession_team.id"].unique()

# DF for Team 1:
pass_map_df_team1 = eventPassdf[(eventPassdf["match_id"] == 303470)
                                & (eventPassdf["possession_team.id"] == team_ids[0])]

pitch = mplsoccer.Pitch(pitch_color='#101010', line_color='white')
fig, ax = pitch.draw(figsize=(16, 8))
arrows = pitch.arrows(pass_map_df_team1["startX"], pass_map_df_team1["startY"],
                      pass_map_df_team1["endX"], pass_map_df_team1["endY"],
                      ax=ax,
                      width=1,
                      color="green")
ax.set_title(pass_map_df_team1["possession_team.name"].unique().item(), fontsize=30)


# DF for Team 2:
pass_map_df_team2 = eventPassdf[(eventPassdf["match_id"] == 303470)
                                & (eventPassdf["possession_team.id"] == team_ids[1])]

pitch = mplsoccer.Pitch(pitch_color='#101010', line_color='white')
fig, ax = pitch.draw(figsize=(16, 8))
arrows = pitch.arrows(pass_map_df_team2["startX"], pass_map_df_team2["startY"],
                      pass_map_df_team2["endX"], pass_map_df_team2["endY"],
                      ax=ax,
                      width=1,
                      color="#ba4f45")
ax.set_title(pass_map_df_team2["possession_team.name"].unique().item(), fontsize=30)


"""## Side-by-Side Pass Maps"""

""" Set Pitch Parameters """
pitch = mplsoccer.Pitch(pitch_color='#101010', line_color='white')
""" Create 1x2 grid of subplots for Analysing two teams """

## here axs is a list of vavlues since nrows and ncols has been provided ##
fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(16, 10))

""" Team 1 Pass Map """
# Draw arrows to create pass map
arrows = pitch.arrows(pass_map_df_team1["startX"],
                      pass_map_df_team1["startY"],
                      pass_map_df_team1["endX"],
                      pass_map_df_team1["endY"],
                      ax=axs[0],
                      width=1,
                      color="green")
# Set title for subplot
axs[0].set_title(pass_map_df_team1["possession_team.name"].unique().item(),
                 fontsize=30)

""" Team 2 Pass Map """
# Draw arrows to create pass map
arrows = pitch.arrows(pass_map_df_team2["startX"],
                      pass_map_df_team2["startY"],
                      pass_map_df_team2["endX"],
                      pass_map_df_team2["endY"],
                      ax=axs[1],
                      width=1,
                      color="#ba4f45")
# Set title for subplot
axs[1].set_title(pass_map_df_team2["possession_team.name"].unique().item(),
                 fontsize=30)



"""
Create grouped data for every match
"""
groups = eventPassdf.groupby(["match_id"])

for name, group in groups:
    print(name)
    print(group.head()) ## will give the remainder of the data as a dictionary ##
    break

"""
Following code will give pass maps for every match
To Save all the plots to a pdf, uncomment line 363-365, 408, 410
"""

#import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
#my_pdf = PdfPages("./reports/passMapsLaLiga2019_20.pdf")
'''
for name, group in groups:
#     print(name)
    team_ids = group["possession_team.id"].unique()
    # Bifurcate team-wise data #
    # print(group.head())
    pass_map_df_team1 = deepcopy(group[(group["possession_team.id"] == team_ids[0])])
    pass_map_df_team2 = deepcopy(group[(group["possession_team.id"] == team_ids[1])])
    
    ## Here onwards the code is picked up from above ##
    ## Ideally a function ccan be created for this purpose ##
    
    # Create 1x2 grid of subplots for Analysing two teams #
    fig, axs = pitch.draw(nrows=1, ncols=2, figsize=(16, 10))

    # Team 1 Pass Map #
    # Draw arrows to create pass map
    arrows = pitch.arrows(pass_map_df_team1["startX"],
                          pass_map_df_team1["startY"],
                          pass_map_df_team1["endX"],
                          pass_map_df_team1["endY"],
                          ax=axs[0],
                          width=1,
                          color="#ba4f45")
    # Set title for subplot
    axs[0].set_title(pass_map_df_team1["possession_team.name"].unique().item(),
                     fontsize=30)

    # Team 2 Pass Map #
    # Draw arrows to create pass map
    arrows = pitch.arrows(pass_map_df_team2["startX"],
                          pass_map_df_team2["startY"],
                          pass_map_df_team2["endX"],
                          pass_map_df_team2["endY"],
                          ax=axs[1],
                          width=1,
                          color="#ad993c")
    # Set title for subplot
    axs[1].set_title(pass_map_df_team2["possession_team.name"].unique().item(),
                     fontsize=30)
    
    fig.show()
    #my_pdf.savefig()

#my_pdf.close()

'''

barcaData = eventPassdf[eventPassdf["team.id"] == 217]

## This will give a count of passes, as seen index will be passer and columns will be receivers ##
## count of type.id gives the number of passes ##
## can use player.id and pass.recipient.id to avoid any challenges arising due to similar names ##

barcaPassMatrix = barcaData.pivot_table(values="type.id", index="player.name",
                                        columns="pass.recipient.name", aggfunc="count")

## following will highlight max value for each column i.e. axis=0 *max no. of passes received ##
## axis = 1, will give max value for each row *max no. of passes created  ##
## here axis properties are opposite than normal  ## 

barcaPassMatrix.style.highlight_max(axis=0).set_precision(0)

## interpret results based on color shading, by default axis = 0 i.e. columns ##s
barcaPassMatrix.style\
    .background_gradient(cmap="Blues")\
    .highlight_null('black').set_precision(0)


## following snippet gives the entire df view that is interactive on hovering ##
"""
# Go to [pandas styling page](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html) and search for "Fun Stuff
## td tabledata, tr tablerows, th tableheaders ##
barcaPassMatrix.style\
    .background_gradient(cmap="Blues", axis=1)\
    .highlight_null('black').set_precision(0)\
    .set_table_styles([dict(selector="th",
                            props=[("font-size", "4pt")]),
                       dict(selector="td",
                            props=[('padding', "0em 0em")]),
                       dict(selector="th:hover",
                            props=[("font-size", "12pt")]),
                       dict(selector="tr:hover td:hover",
                            props=[('max-width', '200px'),
                                   ('font-size', '15pt')])])

## following heatmap using seaborn library, pandas gives option for row, col but sns acccounts for all values in df ##

plt.figure(figsize=(18, 12))
sns.heatmap(data=barcaPassMatrix, annot=True, linewidths=1, fmt=".0f", cmap="viridis")
plt.xlabel("")
plt.ylabel("") """

### El-Classico


eventPassdf.loc[eventPassdf["team.name"].str.contains("Madrid"), ["team.name", "team.id"]].drop_duplicates()

## seasonMetaDataLaLiga2019 contains data for multiple seasons hence season_id is required ##
seasonMetaDataLaLiga2019[["season.season_id", "season.season_name"]].drop_duplicates()
seasonMetaDataLaLiga2019[(seasonMetaDataLaLiga2019["season.season_id"] == 42) &
                         (seasonMetaDataLaLiga2019["home_team.home_team_id"].isin([217, 220])) &
                         (seasonMetaDataLaLiga2019["away_team.away_team_id"].isin([217, 220]))]

#eventPassdf[eventPassdf["match_id"].isin([303596, 303470])]["match_id"].drop_duplicates()
#eventPassdf[eventPassdf["match_id"].isin([303596, 303470])]["match_id"].drop_duplicates()
elClassicoData = eventPassdf[eventPassdf["match_id"].isin([303596, 303470])]

elClassicoPassMatrix = elClassicoData.pivot_table(values="type.id", index="player.name",
                                                  columns="pass.recipient.name", aggfunc="count")

elClassicoPassMatrix.style\
    .background_gradient(cmap="Blues", axis=1)\
    .highlight_null('black').set_precision(0)\
    .set_table_styles([dict(selector="th",
                            props=[("font-size", "4pt")]),
                       dict(selector="td",
                            props=[('padding', "0em 0em")]),
                       dict(selector="th:hover",
                            props=[("font-size", "12pt")]),
                       dict(selector="tr:hover td:hover",
                            props=[('max-width', '200px'),
                                   ('font-size', '15pt')])])
'''
plt.figure(figsize=(18, 12))
sns.heatmap(data=elClassicoPassMatrix, annot=True, linewidths=1, fmt=".0f", cmap="YlGnBu")
plt.xlabel("")
plt.ylabel("")
'''

## checking data for players having passes a minimum of 10 passes ##
plt.figure(figsize=(18, 12))
sns.heatmap(data=elClassicoPassMatrix[elClassicoPassMatrix.apply(lambda x: x > 10)],
            annot=True, linewidths=1, fmt=".0f", cmap="YlGnBu")
plt.xlabel("")
plt.ylabel("")