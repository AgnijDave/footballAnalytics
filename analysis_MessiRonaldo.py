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

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Arc
from plotly.subplots import make_subplots

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

    
## Exploratory Data Analysis 1 - season wise meta data

# Numerical EDA
    
season_wise_meta_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
season_wise_meta_df.info()
season_wise_meta_df = season_wise_meta_df.apply(pd.to_numeric, errors='ignore').round(2)
season_wise_meta_df.describe().round(1) 

season_wise_meta_df[(season_wise_meta_df['season']==2018) & (season_wise_meta_df['Player']=='Messi')]['goals']
    
season_wise_meta_df.groupby(["Player"]).describe().round(1)
season_wise_meta_df.groupby(["Player", "team"])["goals"].sum()

# Visual EDA

num_cols = season_wise_meta_df.select_dtypes(exclude=[object]).columns
season_wise_meta_df.groupby(["Player"])[num_cols].sum()
total_df = season_wise_meta_df.groupby(["Player"])[num_cols].sum().round(1).reset_index()

num_cols.__len__()

'''
import plotly.io as pio
pio.renderers.default='browser'

pio.renderers.default='svg'
'''
 ## Multiple for every column, Interactive Bar graphs giving a comparison of Messi and Ronaldo on sum of the stats collected ##

i = j = 1
# Create an empty subplot:
fig = make_subplots(rows=4, cols=4,
                    shared_xaxes=False,
                    vertical_spacing=0.1,
                    subplot_titles=num_cols)
# Add bar plot for goals conceded in all subplots:
for col in num_cols:
    fig.add_trace(go.Bar(x=total_df["Player"],
                        y=total_df[col],
                        text=total_df[col],
                        textposition="inside",
                        name=col),
                row=i, col=j)
    j += 1
    if j > 4:
        j = 1
        i += 1
    if i > 4:
        i = 1
    break
fig.update_layout(height=800)
fig.show()

'''
xGChain --> Calculating xG for possessions that lead to a shot where the player was involved at least once in that possession, **INCLUDING** the final pass or the shot. \\
xGBuildup --> Calculating xG for possessions that lead to a shot where the player was involved at least once in that possession, **EXCLUDING** the final pass or the shot. \\
'''

fig = None
## Multiple for every column, Interactive Bar graphs giving a comparison of Messi and Ronaldo on a season to season basis from 2014-2021 on sum of the stats collected ##

i = j = 1
# Create an empty subplot:
fig = make_subplots(rows=4, cols=4,
                    shared_xaxes=False,
                    vertical_spacing=0.1,
                    subplot_titles=num_cols)
# Add bar plot for goals conceded in all subplots:
for col in num_cols:
    fig.add_trace(go.Bar(x=season_wise_meta_df["season"],
                         y=season_wise_meta_df[col],
                         text=season_wise_meta_df[col],
                         textposition="inside",
                         name=col),
                row=i, col=j)
    j += 1
    if j > 4:
        j = 1
        i += 1
    if i > 4:
        i = 1
    break
fig.update_layout(height=800) #, barmode="stack")
fig.show()

## Exploratory Data Analysis 2 - shots data

# Numerical EDA

shots_df.info()
shots_df = shots_df.apply(pd.to_numeric, errors="ignore")
shots_df = shots_df.round(2)
shots_df.describe().round(1)

shots_df.groupby(["Player"]).describe().round(1)
shots_df.groupby(["Player", "season"]).describe().round(1)

# Visual EDA

shots_df.head()

## Column Wise segregation into bar graphs for each player
  
px.histogram(data_frame=shots_df, x="result", color="Player",
             barmode="group", title="Shot Result Comparison",
             labels={"result": "", "count": ""})

px.histogram(data_frame=shots_df, x="situation", color="Player",
             barmode="group", title="Shot Play-Pattern Comparison",
             labels={"situation": "", "count": ""})

px.histogram(data_frame=shots_df, x="player_assisted", color="Player",
             barmode="group", title="Assisting Player Comparison",
             labels={"result": "", "count": ""})

"""The Data proves their efficient relatioship with their wing backs"""

px.histogram(data_frame=shots_df, x="h_a", color="Player",
             barmode="group", title="Home-Away Comparison (Shots)",
             labels={"result": "", "count": ""})

px.histogram(data_frame=shots_df[shots_df["result"] == "Goal"], x="h_a", color="Player",
             barmode="group", title="Home-Away Comparison (Goals)",
             labels={"result": "", "count": ""})

px.histogram(data_frame=shots_df[shots_df["result"] == "MissedShots"], x="h_a", color="Player",
             barmode="group", title="Home-Away Comparison (Goals)",
             labels={"result": "", "count": ""})



## Comparative Analysis
## Goals vs Shots (per season)

## Scatter plot to show shots vs goals

fig = px.scatter(x="shots", y="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season", size="npg",
                 opacity=.8)
fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(showlegend=True, height=600, width=1200)
fig.show()

##
## trace across average goals to min max of shots ## ## 26 goals a season, taking anywhere bw 1 to 232 shots a season
fig = px.scatter(x="shots", y="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season", size="npg",
                 opacity=.8)
fig.add_trace(go.Scatter(x=[season_wise_meta_df["shots"].min(), season_wise_meta_df["shots"].max() + 5],
                         y=[season_wise_meta_df["goals"].mean(), season_wise_meta_df["goals"].mean()],
                         name="Avg. Goals"))
fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(showlegend=True, height=800, width=1500)
fig.show()

## trace across average shots to min max of goals ## ## 158 shots a season, scoring anywhere between 0 to 58 goals
fig = px.scatter(x="shots", y="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season", size="npg",
                 opacity=.8)
fig.add_trace(go.Scatter(x=[season_wise_meta_df["shots"].mean(), season_wise_meta_df["shots"].mean()],
                         y=[season_wise_meta_df["goals"].min(), season_wise_meta_df["goals"].max() + 5],
                         name="Avg. Shots"))
fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(showlegend=True, height=800, width=1500)
fig.show()

## both the above traces
## expect 26 goals a season, taking anywhere bw 1 to 232 shots a season
## expect 158 shots a season, scoring anywhere between 0 to 58 goals

## Ronaldo seems to win this battle with max stats on top right corner of graph (shots vs goals)

fig = px.scatter(x="shots", y="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season", size="npg",
                 opacity=.8)
fig.add_trace(go.Scatter(x=[season_wise_meta_df["shots"].min(), season_wise_meta_df["shots"].max() + 5],
                         y=[season_wise_meta_df["goals"].mean(), season_wise_meta_df["goals"].mean()],
                         name="Avg. Goals"))
fig.add_trace(go.Scatter(x=[season_wise_meta_df["shots"].mean(), season_wise_meta_df["shots"].mean()],
                         y=[season_wise_meta_df["goals"].min(), season_wise_meta_df["goals"].max() + 5],
                         name="Avg. Shots"))
fig.update_traces(textposition='top center', textfont_size=10)
fig.update_layout(showlegend=True, height=800, width=1500)
fig.show()
##

"""## Total Goals vs Non-Penalty Goals (per season)"""

## 26 goals on average from which 22 can be expected to be non penalty goals npg -- Ronaldo gets the highest record although Messi is more consistent
fig = px.scatter(y="goals", x="npg", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season",
                 opacity=.8)
fig.add_trace(go.Scatter(x=[season_wise_meta_df["npg"].min(), season_wise_meta_df["npg"].max() + 5],
                         y=[season_wise_meta_df["goals"].mean(), season_wise_meta_df["goals"].mean()],
                         name="Avg. Goals"))
fig.add_trace(go.Scatter(x=[season_wise_meta_df["npg"].mean(), season_wise_meta_df["npg"].mean()],
                         y=[season_wise_meta_df["goals"].min(), season_wise_meta_df["goals"].max() + 5],
                         name="Avg. npg"))
fig.update_traces(textposition='top center', textfont_size=10, marker=dict(size=10))
fig.update_layout(showlegend=True, height=800, width=1200)
fig.show()

"""## Total Goals vs Total Expected Goals (per season)"""

## Linearly increasing, both out perform every season
fig = px.scatter(y="xG", x="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season",
                 opacity=.8)
fig.add_trace(go.Scatter(x=[0, season_wise_meta_df[["goals", "npg"]].max().max() + 5],
                         y=[0, season_wise_meta_df[["goals", "npg"]].max().max() + 5]))
fig.update_traces(textposition='top center', textfont_size=10, marker=dict(size=10))
fig.update_layout(showlegend=True, height=800, width=1200)
fig.show()

"""## Total Goals vs Total Assists (per season)"""

## Messi with the most assissts in a season whereas Ronaldo has the max goals, no surprise in that!
fig = px.scatter(y="assists", x="goals", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season",
                 opacity=.8)
fig.add_trace(go.Scatter(x=season_wise_meta_df["goals"],
                         y=[season_wise_meta_df["assists"].mean()]*len(season_wise_meta_df),
                         name="Avg. Assists",
                         marker={"size": .1}))
fig.add_trace(go.Scatter(x=[season_wise_meta_df["goals"].mean()]*len(season_wise_meta_df),
                         y=season_wise_meta_df["assists"],
                         name="Avg. Goals",
                         marker={"size": .1}))
fig.update_traces(textposition='top center',
                  textfont_size=10,
                  marker=dict(size=10))
fig.update_layout(showlegend=True, height=800, width=1200)
fig.show()

"""## Total Expected Goals (xG) vs Total Expected Assists (xA) (per season)"""

## Same observation as above.
fig = px.scatter(y="xA", x="xG", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season",
                 opacity=.8)
fig.add_trace(go.Scatter(x=season_wise_meta_df["xG"],
                         y=[season_wise_meta_df["xA"].mean()]*len(season_wise_meta_df),
                         name="Avg. xA",
                         marker={"size": .1}))
fig.add_trace(go.Scatter(x=[season_wise_meta_df["xG"].mean()]*len(season_wise_meta_df),
                         y=season_wise_meta_df["xA"],
                         name="Avg. xG",
                         marker={"size": .1}))
fig.update_traces(textposition='top center',
                  textfont_size=10,
                  marker=dict(size=10))
fig.update_layout(showlegend=True, height=800, width=1200)
fig.show()

"""## Total xGChain vs Total xGBuildup (per season)"""
'''
xGChain --> Calculating xG for possessions that lead to a shot where the player was involved at least once in that possession, **INCLUDING** the final pass or the shot. \\
xGBuildup --> Calculating xG for possessions that lead to a shot where the player was involved at least once in that possession, **EXCLUDING** the final pass or the shot. \\
'''

## As exoected Messi is a more wholesome player in terms of play and thus prevalent in the top right corner, Ronaldo is attack minded
fig = px.scatter(x="xGChain", y="xGBuildup", data_frame=season_wise_meta_df,
                 hover_name="season", text="Player", symbol="season",
                 opacity=.8)
fig.add_trace(go.Scatter(x=season_wise_meta_df["xGChain"],
                         y=[season_wise_meta_df["xGBuildup"].mean()]*len(season_wise_meta_df),
                         name="Avg. xGBuildup",
                         marker={"size": .1}))
fig.add_trace(go.Scatter(x=[season_wise_meta_df["xGChain"].mean()]*len(season_wise_meta_df),
                         y=season_wise_meta_df["xGBuildup"],
                         name="Avg. xGChain",
                         marker={"size": .1}))

fig.update_traces(textposition='top center',
                  textfont_size=10,
                  marker=dict(size=10))
fig.update_layout(showlegend=True, height=800, width=1200)
fig.show()



# Shot Analysis
## Function for creating pitch map

# Setting boundaries and midpoint:
x_lims = [0, 1.15]
y_lims = [0, 0.74]

x_mid = x_lims[1]/2
y_mid = y_lims[1]/2

# Setting color and linewidth:
background_color = "black"
line_color = "white"
line_width = 2.

def create_full_pitch(x_lims, y_lims, background_color="white", line_color="black", line_width=2.):
    """
    Function to create a full-scale pitch based on input dimensions
    :params:
    x_lims: min and max limits for the length of the field
    y_lims: min and max limits for the width/breadth of the field
    background_color: Background color of the field
    line_color: Color for all the lines in the field (Keep this color in contrast with background_color for optimal visual results)
    line_width: The thickness of the outer and center lines
    """
    # Create figure:
    fig = plt.figure(facecolor=background_color, figsize=(12, 7))
    ax = fig.add_subplot(111, facecolor=background_color)

    # Pitch Outline & Centre Line
    plt.plot([x_lims[0], x_lims[0]], [y_lims[0], y_lims[1]], linewidth=line_width, color=line_color)  # left goal-line
    plt.plot([x_lims[0], x_lims[1]], [y_lims[1], y_lims[1]], linewidth=line_width, color=line_color)  # Upper side-line
    plt.plot([x_lims[1], x_lims[1]], [y_lims[1], y_lims[0]], linewidth=line_width, color=line_color)  # Right goal-line
    plt.plot([x_lims[1], x_lims[0]], [y_lims[0], y_lims[0]], linewidth=line_width, color=line_color)  # Lower side-line
    plt.plot([x_mid, x_mid], [y_lims[0], y_lims[1]], linewidth=line_width, color=line_color)  # Center line

    # Left Penalty Area
    plt.plot([x_lims[0]+.18, x_lims[0]+.18], [y_mid - .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[0], x_lims[0]+.18], [y_mid + .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[0], x_lims[0]+.18], [y_mid - .22, y_mid - .22], color=line_color)

    # Right Penalty Area
    plt.plot([x_lims[1] - .18, x_lims[1] - .18], [y_mid - .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .18], [y_mid + .22, y_mid + .22], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .18], [y_mid - .22, y_mid - .22], color=line_color)

    # Left 6yd box Area
    plt.plot([x_lims[0]+.06, x_lims[0]+.06], [y_mid - .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[0], x_lims[0]+.06], [y_mid + .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[0], x_lims[0]+.06], [y_mid - .06, y_mid - .06], color=line_color)

    # # Right 6yd box Area
    plt.plot([x_lims[1] - .06, x_lims[1] - .06], [y_mid - .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .06], [y_mid + .06, y_mid + .06], color=line_color)
    plt.plot([x_lims[1], x_lims[1] - .06], [y_mid - .06, y_mid - .06], color=line_color)

    # Prepare Circles
    centre_circle = plt.Circle((x_mid, y_mid), .1, color=line_color, fill=False)
    centre_spot = plt.Circle((x_mid, y_mid), 0.005, color=line_color)
    left_pen_spot = plt.Circle((x_lims[0]+0.12, y_mid), 0.005, color=line_color)
    right_pen_spot = plt.Circle((x_lims[1] - 0.12, y_mid), 0.005, color=line_color)

    # Draw Circles
    ax.add_patch(centre_circle)
    ax.add_patch(centre_spot)
    ax.add_patch(left_pen_spot)
    ax.add_patch(right_pen_spot)

    # Prepare Arcs
    left_arc = Arc((x_lims[0] + .12, y_mid), height=.183, width=.183, angle=0, theta1=310, theta2=50, color=line_color)
    right_arc = Arc((x_lims[1] - .12, y_mid), height=.183, width=.183, angle=0, theta1=130, theta2=230, color=line_color)

    # Draw Arcs
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    plt.axis("off")

    return ax

"""## Plot Shot-maps"""

## Using the X Y location to get an idea from where in the pitch were the goal attempts along with a heat map ## 

shots_df["X"] = shots_df["X"].multiply(x_lims[1])
shots_df["Y"] = shots_df["Y"].multiply(y_lims[1])

ax1 = create_full_pitch(x_lims, y_lims)
sns.scatterplot(x="X", y="Y", data=shots_df[shots_df["Player"] == "Cristiano"], size="xG", ax=ax1)
ax1.set_xlim([x_mid, x_lims[1]])
ax1.set_ylim(y_lims)

# ---

ax2 = create_full_pitch(x_lims, y_lims)
sns.scatterplot(x="X", y="Y", data=shots_df[shots_df["Player"] == "Messi"], size="xG", ax=ax2)
ax2.set_xlim([x_mid, x_lims[1]])
ax2.set_ylim(y_lims)

"""## Plot HeatMaps"""

ax1 = create_full_pitch(x_lims, y_lims)
sns.kdeplot(x="X", y="Y", data=shots_df[shots_df["Player"] == "Cristiano"], shade=True, n_levels=10, ax=ax1)
ax1.set_xlim([x_mid, x_lims[1]])
ax1.set_ylim(y_lims)

# ---

ax1 = create_full_pitch(x_lims, y_lims)
sns.kdeplot(x="X", y="Y", data=shots_df[shots_df["Player"] == "Messi"], shade=True, n_levels=10, ax=ax1)
ax1.set_xlim([x_mid, x_lims[1]])
ax1.set_ylim(y_lims)



"""# The Final Comparison"""

season_wise_meta_df
radar_df = season_wise_meta_df.groupby(["Player"])[num_cols].sum().reset_index()

radar_df
radar_df.columns

cols_for_radar = ['goals', 'shots', 'xG',
                  'assists', 'xA', 'key_passes',
                  'npg', 'npxG', 'xGChain', 'xGBuildup']

## Radar Plots for Comparison Analysis, Both have a similar plot, for collected stats and expected stats

fig = go.Figure()
# Add Radar plots for different players:
fig.add_trace(go.Scatterpolar(
    r=radar_df.loc[(radar_df["Player"] == "Cristiano"), cols_for_radar].values.flatten(),
    theta=cols_for_radar,
    fill='toself',
    name='Cristiano'))
fig.add_trace(go.Scatterpolar(
    r=radar_df.loc[(radar_df["Player"] == "Messi"), cols_for_radar].values.flatten(),
    theta=cols_for_radar,
    fill='toself',
    name="Messi"))
# Additional properties for the plot:
fig.update_layout(
    title="Cristiano vs Messi",
polar=dict(
    radialaxis=dict(
    visible=True,
    )),
showlegend=True
)
fig.show()

## Per90 Stats

per90Cols = ['goals', 'shots', 'xG',
             'assists', 'xA', 'key_passes',
             'npg', 'npxG', 'xGChain', 'xGBuildup']

for col in per90Cols:
    radar_df[col + "Per90"] = radar_df[col].divide(radar_df["time"]).multiply(90)

cols_for_radar = [i + "Per90" for i in per90Cols]

radar_df[cols_for_radar]

# Initiate the plotly go figure
fig = go.Figure()
# Add Radar plots for different players:
fig.add_trace(go.Scatterpolar(
    r=radar_df.loc[(radar_df["Player"] == "Cristiano"), cols_for_radar].sum(),
    theta=cols_for_radar,
    fill='toself',
    name='Cristiano'))
fig.add_trace(go.Scatterpolar(
    r=radar_df.loc[(radar_df["Player"] == "Messi"), cols_for_radar].sum(),
    theta=cols_for_radar,
    fill='toself',
    name="Messi"))
# Additional properties for the plot:
fig.update_layout(
    title="Cristiano vs Messi",
polar=dict(
    radialaxis=dict(
    visible=True,
    )),
showlegend=True
)
fig.show()