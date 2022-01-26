import pandas as pd
import numpy as np
import glob

files = glob.glob('./data/*.csv')
datasets = { f:pd.read_csv(f) for f in files }

for k,v in datasets.items():
    if 'event' in k:
        eventsDataLaLiga2019 = v
    else:
        seasonMetaDataLaLiga2019 = v
        
seasonCols = ['match_id', 'match_date', 'kick_off', 'home_score', 'away_score',
              'home_team.home_team_id', 'home_team.home_team_name',
              'away_team.away_team_id', 'away_team.away_team_name']
eventsDataLaLiga2019 = pd.merge(eventsDataLaLiga2019,
                                seasonMetaDataLaLiga2019[seasonCols],
                       how="left", on=["match_id"])

'''For Passing Analysis ,type.id = 30, will contain the pass event
eventPassdf = deepcopy(eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 30])
'''

## Feature engineering ##
## 1. Whether that particular event's player started the game

from ast import literal_eval

## type.id = 35, will contain the starting lineup
finalLineUpdf = pd.DataFrame()
for m_id in eventsDataLaLiga2019["match_id"].unique():
    print(m_id, "\n")
    match_df = eventsDataLaLiga2019[(eventsDataLaLiga2019["match_id"] == m_id)
                                    & (eventsDataLaLiga2019["type.id"] == 35)]["tactics.lineup"].apply(literal_eval)
    
    df1 = pd.json_normalize(match_df.iloc[0])
    df2 = pd.json_normalize(match_df.iloc[1])

    df = df1.append(df2)
    df.insert(0, "match_id", m_id)
    
    df["started"] = "Yes"
    
    finalLineUpdf = finalLineUpdf.append(df)
    
    
eventsDataLaLiga2019 = pd.merge(eventsDataLaLiga2019,
                                finalLineUpdf[['match_id', 'player.id', 'jersey_number', 'started']],
                                how="left",
                                on=['match_id', 'player.id'])

## 2. Assign total minutes played for that player, using substitution data + started column from above -
##    --> minMinute to last event for that player will give the total.

''' inaccurate approach
minsPlayeddf = eventsDataLaLiga2019.groupby(["match_id", "player.id"]).agg(
    {"player.name": "first", "team.name": "first", "minute": ["min", "max"]}).reset_index()'''

# following will give the results where substitution replacement id is not null
subData = eventsDataLaLiga2019[["match_id", "substitution.replacement.id", "minute"]]\
.dropna(subset=["substitution.replacement.id"])

subData.rename(columns={"minute": "minMinute",
                        "substitution.replacement.id": "subID"}, inplace=True)
    
eventsDataLaLiga2019 = pd.merge(eventsDataLaLiga2019, subData, how="left",
                       left_on=["match_id", "player.id"],
                       right_on=["match_id", "subID"])

## assertion, all nan values should be present
assert str(eventsDataLaLiga2019.dropna(subset=["started"])["minMinute"].unique()[0]) == 'nan'

subOutData = eventsDataLaLiga2019[eventsDataLaLiga2019['substitution.outcome.id'].notnull()]\
            [["match_id", "player.id", "minute"]]
subOutData.rename(columns={"minute": "maxMinute"}, inplace=True)

eventsDataLaLiga2019 = pd.merge(eventsDataLaLiga2019, subOutData,
         how="left",
         on=["match_id", "player.id"])

eventsDataLaLiga2019["maxMinsMatch"] =\
    eventsDataLaLiga2019.groupby(["match_id"])["minute"].transform(lambda x: x.max())
    
eventsDataLaLiga2019["minMinute"] = np.where(eventsDataLaLiga2019["started"] == "Yes",
                                             0, eventsDataLaLiga2019["minMinute"])

#temp_df = eventsDataLaLiga2019[0:50]

eventsDataLaLiga2019["maxMinute"] = np.where(eventsDataLaLiga2019["maxMinute"].isnull(),
                                             eventsDataLaLiga2019["maxMinsMatch"],
                                             eventsDataLaLiga2019["maxMinute"])

#temp_df_ = eventsDataLaLiga2019[0:50]

minsPlayeddf = eventsDataLaLiga2019.groupby(["match_id", "player.id"]).agg(
    {"player.name": "first", "team.name": "first",
     "minMinute": "first", "maxMinute": "max"}).reset_index()
    
minsPlayeddf["minsPlayed"] = minsPlayeddf["maxMinute"].subtract(minsPlayeddf["minMinute"])
    
eventsDataLaLiga2019 = pd.merge(eventsDataLaLiga2019,
                       minsPlayeddf[["match_id", "player.id", "minsPlayed"]],
                       how="left",
                       on=["match_id", "player.id"])

eventsDataLaLiga2019.to_csv("./data/matchwise_events_data_updated.csv", index=False)