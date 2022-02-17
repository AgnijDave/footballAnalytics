import pandas as pd
import numpy as np

eventPassDataLaLiga2019 = pd.read_csv('../data/eventPassData.csv', low_memory=False)


## Analysis on player Vision 
## Event level data has info about pass.through_ball, pass.shot_assisst, pass.goal_assisst

playerWisedf = eventPassDataLaLiga2019.groupby(["player.id"]).agg({"player.name": "first",
                                                                   "team.name": "first",
                                                                   "type.id": "count",
                                                                   "pass.through_ball": "sum",
                                                                   "pass.shot_assist": "sum",
                                                                   "pass.goal_assist": "sum"})

playerMatchMinsdf = eventPassDataLaLiga2019.drop_duplicates(subset=["player.id", "match_id"]).groupby(["player.id"])\
.agg({"minsPlayed": "sum"}) ## "player.name": "first" -- to check concat is correct

playerWisedf = pd.concat([playerWisedf, playerMatchMinsdf], axis=1)

playerWisedf.rename(columns={"type.id": "totalPasses",
                             "pass.through_ball": "passTB",
                             "pass.shot_assist": "passSA",
                             "pass.goal_assist": "passGA"},
                    inplace=True)

"""### Calculate Per90 Since dataset is mostly for Barcelona matches thus to get a fair sense of other players,
also only Barcelona team players 
    will have credible analysis ###"""

per90Cols = ["totalPasses", "passTB", "passSA", "passGA"]

for col in per90Cols:
    playerWisedf[col + "Per90"] = playerWisedf[col].divide(playerWisedf["minsPlayed"]).multiply(90)
    

"""### Vision Ratings ###"""

playerWisedf["visionRating"] =\
    (playerWisedf["passTBPer90"]*0.2)\
        .add(playerWisedf["passSAPer90"]*0.3)\
        .add(playerWisedf["passGAPer90"]*0.5)

## get rank of top 10 players ##
playerWisedf.loc[playerWisedf["visionRating"].nlargest(10).index, ["player.name", "team.name", "visionRating"]]
## playerWisedf.loc[3122]

## get a more credible dataset to compare, players having played more than 45 minutes ##
playerWiseFiltereddf = playerWisedf[(playerWisedf["minsPlayed"] > 45)]

playerWiseFiltereddf.loc[playerWiseFiltereddf["visionRating"].nlargest(10).index]\
[["player.name", "team.name", "visionRating"]]

## scaling between standard 1-10 ## 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((1, 10))
playerWiseFiltereddf["visionRating"] =\
    scaler.fit_transform(np.array(playerWiseFiltereddf["visionRating"]).reshape(-1, 1))

playerWiseFiltereddf["visionRating"].max()

playerWiseFiltereddf.loc[playerWiseFiltereddf["visionRating"].nlargest(10).index]\
[["player.name", "team.name", "visionRating"]]



"""### Ball Control ###"""

eventsDataLaLiga2019 = pd.read_csv('../data/matchwise_events_data_updated.csv')


eventsDataLaLiga2019[["type.id", "type.name"]].drop_duplicates()

"""Data Prep - Carry"""

carryDataLaLiga1920 = eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 43]

'''carryDataLaLiga1920.groupby(["player.id"]).agg({"player.name": "first",
                                                    "team.name": "first",
                                                    "type.id": "count"})\
.sort_values("type.id", ascending=False)'''

carryPlayerData = carryDataLaLiga1920.groupby(["player.id"]).agg({"player.name": "first",
                                                                  "team.name": "first",
                                                                  "type.id": "count"})
carryPlayerData.rename(columns={"type.id": "totCarry"}, inplace=True)

#pd.concat([carryPlayerData, playerMatchMinsdf], axis=1) ## will give a lot of null values
#pd.merge(carryPlayerData, playerMatchMinsdf, how="left", left_index=True, right_index=True)

## add True for both left and right indexes means joining on index ##

carryPlayerData = pd.merge(carryPlayerData, playerMatchMinsdf, how="left",
                           left_index=True, right_index=True)
carryPlayerData["totCarryPer90"] = carryPlayerData["totCarry"].divide(carryPlayerData["minsPlayed"])*90

"""Data Prep - Dispossessed"""

dispDataLaLiga1920 = eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 3]
dispPlayerData = dispDataLaLiga1920.groupby(["player.id"]).agg({"player.name": "first",
                                                                  "team.name": "first",
                                                                  "type.id": "count"})

dispPlayerData.rename(columns={"type.id": "totDisp"}, inplace=True)
dispPlayerData = pd.merge(dispPlayerData, playerMatchMinsdf, how="left",
                          left_index=True, right_index=True)

dispPlayerData["totDispPer90"] = dispPlayerData["totDisp"].divide(dispPlayerData["minsPlayed"])*90

## Since we will Merge all 4 entities ##
dispPlayerData.drop(columns=['player.name', 'team.name', 'minsPlayed'], inplace=True)

"""Data Prep - Dribble"""

dribbleDataLaLiga1920 = eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 14]

dribblePlayerData = dribbleDataLaLiga1920.groupby(["player.id"]).agg({"player.name": "first",
                                                                  "team.name": "first",
                                                                  "type.id": "count"})

dribblePlayerData.rename(columns={"type.id": "totDribble"}, inplace=True)
dribblePlayerData = pd.merge(dribblePlayerData, playerMatchMinsdf, how="left", left_index=True, right_index=True)

dribblePlayerData["totDribblePer90"] = dribblePlayerData["totDribble"].divide(dribblePlayerData["minsPlayed"])*90


dribblePlayerData.loc[dribblePlayerData["totDribblePer90"].nlargest(10).index,
                      ["player.name", "team.name", "totDribblePer90"]]
dribblePlayerData.drop(columns=['player.name', 'team.name', 'minsPlayed'], inplace=True)

"""Data Prep - Miscontrol"""

miscontrolDataLaLiga1920 = eventsDataLaLiga2019[eventsDataLaLiga2019["type.id"] == 38]

miscontrolPlayerData = miscontrolDataLaLiga1920.groupby(["player.id"]).agg({"player.name": "first",
                                                                  "team.name": "first",
                                                                  "type.id": "count"})
miscontrolPlayerData.rename(columns={"type.id": "totMiscontrol"}, inplace=True)

miscontrolPlayerData = pd.merge(miscontrolPlayerData, playerMatchMinsdf,
                                how="left",
                                left_index=True, right_index=True)

miscontrolPlayerData["totMiscontrolPer90"] =\
    miscontrolPlayerData["totMiscontrol"].divide(miscontrolPlayerData["minsPlayed"])*90

## nsmallest since lesser miscontrol indicates better quality of touch ##
miscontrolPlayerData.loc[miscontrolPlayerData["totMiscontrolPer90"].nsmallest(10).index,
                         ["player.name", "team.name", "totMiscontrolPer90"]]
miscontrolPlayerData.drop(columns=['player.name', 'team.name', 'minsPlayed'], inplace=True)



""" Ball Control Ratings
Get Ball Control Data   """

ballControlData = pd.merge(carryPlayerData, dispPlayerData,
                           how="outer", left_index=True, right_index=True)
ballControlData = pd.merge(ballControlData, dribblePlayerData,
                           how="outer", left_index=True, right_index=True)
ballControlData = pd.merge(ballControlData, miscontrolPlayerData,
                           how="outer", left_index=True, right_index=True)

"""  Using Non-Standardized weights: as we want to reward and penalize some actions
Ball_Control_Rating = (totCarryPer90 * 2) + (totDispPer90 * -1) + (totDribblePer90 * 1) + (totMiscontrolPer90 * -2)
"""

### get an idea of ball control data ###
ballControlData["bcRating"] =\
    (ballControlData["totCarryPer90"]*2)\
        .add(ballControlData["totDispPer90"]*-1)\
        .add(ballControlData["totDribblePer90"]*1)\
        .add(ballControlData["totMiscontrolPer90"]*-2)

ballControlData.loc[ballControlData["bcRating"].nlargest(20).index]\
[["player.name", "team.name", "bcRating"]]
###

## Fill Nan values with 0 to avoid any errors ##
ballControlData[["totCarryPer90", "totDispPer90", "totDribblePer90", "totMiscontrolPer90"]]=\
    ballControlData[["totCarryPer90", "totDispPer90", "totDribblePer90", "totMiscontrolPer90"]].fillna(0)

ballControlData["bcRating"] =\
    (ballControlData["totCarryPer90"]*2)\
        .add(ballControlData["totDispPer90"]*-1)\
        .add(ballControlData["totDribblePer90"]*1)\
        .add(ballControlData["totMiscontrolPer90"]*-2)

ballControlData["bcRating"] =\
    scaler.fit_transform(np.array(ballControlData["bcRating"]).reshape(-1, 1))

ballControlData.loc[ballControlData["bcRating"].nlargest(20).index]\
[["player.name", "team.name", "bcRating"]]


""" Creativity Ratings = (totCarryPer90 * 0.05) + (passTBPer90 * 0.2) + (totDribblePer90 * 0.1)  + (passSAPer90 * 0.25) + (passGAPer90 * 0.4) """


carryPlayerData.drop(columns=['player.name', 'team.name', 'minsPlayed'], inplace=True)

creativityData = pd.merge(playerWisedf, dribblePlayerData,
                           how="outer", left_index=True, right_index=True)
creativityData = pd.merge(creativityData, carryPlayerData,
                           how="outer", left_index=True, right_index=True)

creativityData.isnull().sum()

creativityData[["totCarryPer90", "passTBPer90", "totDribblePer90", "passSAPer90", "passGAPer90"]] =\
    creativityData[["totCarryPer90", "passTBPer90", "totDribblePer90", "passSAPer90", "passGAPer90"]].fillna(0)

creativityData["creativityRating"] =\
    (creativityData["totCarryPer90"]*0.05)\
        .add(creativityData["passTBPer90"]*0.2)\
        .add(creativityData["totDribblePer90"]*0.1)\
        .add(creativityData["passSAPer90"]*0.25)\
        .add(creativityData["passGAPer90"]*0.4)

creativityData["creativityRating"] =\
    scaler.fit_transform(np.array(creativityData["creativityRating"]).reshape(-1, 1))

creativityData.loc[creativityData["creativityRating"].nlargest(10).index]\
[["player.name", "team.name", "creativityRating"]]


""" Passing Ability
(Accuracy,  Ground Pass Accuracy,  Low Pass Accuracy,  High Pass Accuracy,  Miscommunication,  Under Pressure Accuracy,  Through Ball Accuracy)

### Total Passing Data ### 
"""

## null values in pass.outcome.id indicate complete passes ##

totPassData = eventPassDataLaLiga2019.groupby(["player.id"]).agg({"player.name": "first",
                                                    "team.name": "first",
                                                    "type.id": "count",
                                                    "pass.outcome.id": lambda x: (x.isnull()).sum()})

totPassData.rename(columns={"type.id": "totPasses",
                            "pass.outcome.id": "succPasses"},
                  inplace=True)

totPassData["passAccuracy"] = totPassData["succPasses"].divide(totPassData["totPasses"])


## visualization for accuracy distribution in bins ##

binList = []
for i in range(0, 101, 10):
    binList.append(i/100)

totPassData["passAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])

"""### Ground Pass Data"""

# eventPassDataLaLiga2019[["pass.height.id", "pass.height.name"]].drop_duplicates()

gpData = eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.height.id"] == 1]
gpPlayerData = gpData.groupby(["player.id"]).agg({"type.id": "count",
                                                  "pass.outcome.id": lambda x: (x.isnull()).sum()})

gpPlayerData.rename(columns={"type.id": "totGPasses",
                             "pass.outcome.id": "succGPasses"},
                    inplace=True)

gpPlayerData["gpAccuracy"] = gpPlayerData["succGPasses"].divide(gpPlayerData["totGPasses"])

""" Accuracy Distribution"""
gpPlayerData["gpAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])


"""### Low Pass Data"""

lpData = eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.height.id"] == 2]
lpPlayerData = lpData.groupby(["player.id"]).agg({"type.id": "count",
                                                  "pass.outcome.id": lambda x: (x.isnull()).sum()})

lpPlayerData.rename(columns={"type.id": "totLPasses",
                             "pass.outcome.id": "succLPasses"},
                    inplace=True)

lpPlayerData["lpAccuracy"] = lpPlayerData["succLPasses"].divide(lpPlayerData["totLPasses"])


"""Accuracy Distribution"""
lpPlayerData["lpAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])


"""### High Pass Data"""

hpData = eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.height.id"] == 3]
hpPlayerData = hpData.groupby(["player.id"]).agg({"type.id": "count",
                                                  "pass.outcome.id": lambda x: (x.isnull()).sum()})

hpPlayerData.rename(columns={"type.id": "totHPasses",
                             "pass.outcome.id": "succHPasses"},
                    inplace=True)

hpPlayerData["hpAccuracy"] = hpPlayerData["succHPasses"].divide(hpPlayerData["totHPasses"])

"""Accuracy Distribution"""
hpPlayerData["hpAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])


"""### Under Pressure Data"""


upData = eventPassDataLaLiga2019[eventPassDataLaLiga2019["under_pressure"].notnull()]
upPlayerData = upData.groupby(["player.id"]).agg({"type.id": "count",
                                                  "pass.outcome.id": lambda x: (x.isnull()).sum()})

upPlayerData.rename(columns={"type.id": "totPassesUP",
                             "pass.outcome.id": "succPassesUP"},
                    inplace=True)

upPlayerData["upAccuracy"] =\
    upPlayerData["succPassesUP"].divide(upPlayerData["totPassesUP"])

"""Accuracy Distribution"""
upPlayerData["upAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])


"""### Through Ball Data"""

tbData = eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.through_ball"].notnull()]
tbPlayerData = tbData.groupby(["player.id"]).agg({"type.id": "count",
                                   "pass.outcome.id": lambda x: (x.isnull()).sum()})

tbPlayerData.rename(columns={"type.id": "totPassesTB",
                             "pass.outcome.id": "succPassesTB"},
                    inplace=True)

tbPlayerData["tbAccuracy"] =\
    tbPlayerData["succPassesTB"].divide(tbPlayerData["totPassesTB"])

"""Accuracy Distribution"""
tbPlayerData["tbAccuracy"].hist(bins=[i/100 for i in range(0, 101, 10)])
eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.through_ball"].notnull()]["pass.outcome.id"].unique()


"""### Miscommunication Data"""
len(eventPassDataLaLiga2019[eventPassDataLaLiga2019["pass.miscommunication"].notnull()])/len(eventPassDataLaLiga2019)*100


""" Generate Passing Ability Data """
# Join Total Passes df with Ground Passes df:
passingAbilityData = pd.merge(totPassData, gpPlayerData,
                           how="outer", left_index=True, right_index=True)

# Join Final Pass Ability df with Low Passes df:
passingAbilityData = pd.merge(passingAbilityData, lpPlayerData,
                           how="outer", left_index=True, right_index=True)

# Join Final Pass Ability df with High Passes df:
passingAbilityData = pd.merge(passingAbilityData, hpPlayerData,
                           how="outer", left_index=True, right_index=True)

# Join Final Pass Ability df with Under Pressure Passes df:
passingAbilityData = pd.merge(passingAbilityData, upPlayerData,
                           how="outer", left_index=True, right_index=True)

# Join Final Pass Ability df with Through Ball Passes df:
passingAbilityData = pd.merge(passingAbilityData, tbPlayerData,
                           how="outer", left_index=True, right_index=True)

"""Using Standardized weights:
paRating = (passAccuracy * 0.08) + (gpAccuracy * 0.02) + (lpAccuracy * 0.1)  + (hpAccuracy * 0.15) + (upAccuracy * 0.25) + (tbAccuracy * 0.4)
"""

passingAbilityData[['passAccuracy', 'gpAccuracy',
                    'lpAccuracy', 'hpAccuracy',
                    'upAccuracy', 'tbAccuracy']] =\
    passingAbilityData[['passAccuracy', 'gpAccuracy',
                    'lpAccuracy', 'hpAccuracy',
                    'upAccuracy', 'tbAccuracy']].fillna(0)

passingAbilityData["paRating"] =\
    (passingAbilityData["passAccuracy"]*0.05)\
        .add(passingAbilityData["gpAccuracy"]*0.05)\
        .add(passingAbilityData["lpAccuracy"]*0.1)\
        .add(passingAbilityData["hpAccuracy"]*0.15)\
        .add(passingAbilityData["upAccuracy"]*0.25)\
        .add(passingAbilityData["tbAccuracy"]*0.4)

passingAbilityData["paRating"] =\
    scaler.fit_transform(np.array(passingAbilityData["paRating"]).reshape(-1, 1))

passingAbilityData.loc[passingAbilityData["paRating"].nlargest(10).index]\
[["player.name", "team.name", "paRating"]]


"""# Playmaker Ratings  #"""
""" Get Playmaker Data """

# Join Vision  df with Ball Control df:
playmakerDataFinal = pd.merge(playerWisedf[['player.name', 'team.name',
                                            'minsPlayed', 'visionRating']],
                              ballControlData["bcRating"],
                              how="outer",
                              left_index=True, right_index=True)

# Join Final Playmaker df with Creativity df:
playmakerDataFinal = pd.merge(playmakerDataFinal,
                              creativityData["creativityRating"],
                              how="outer",
                              left_index=True, right_index=True)

# Join Final Playmaker df with Passing Ability df:
playmakerDataFinal = pd.merge(playmakerDataFinal,
                              passingAbilityData['paRating'],
                              how="outer",
                              left_index=True, right_index=True)

playmakerDataFinal["playmakerRating"] =\
    playmakerDataFinal[['visionRating', 'bcRating',
                        'creativityRating', 'paRating']].mean(axis=1) # axis 1 implies row wise mean #

playmakerDataFinal.loc[playmakerDataFinal["playmakerRating"].nlargest(10).index]\
    [["player.name", "team.name", "playmakerRating"]]

playmakerDataFinalFiltered = playmakerDataFinal[playmakerDataFinal["minsPlayed"] > 45]

playmakerDataFinalFiltered.loc[
    playmakerDataFinalFiltered["playmakerRating"].nlargest(10).index]\
        [["player.name", "team.name", "playmakerRating"]]

""" playmakerRating = (visionRating * 0.4) + (bcRating * 0.2) + (creativityRating * 0.3)  + (paRating * 0.1) """

playmakerDataFinal["playmakerRating"] =\
    (playmakerDataFinal["visionRating"]*0.4)\
        .add(playmakerDataFinal["bcRating"]*0.2)\
        .add(playmakerDataFinal["creativityRating"]*0.3)\
        .add(playmakerDataFinal["paRating"]*0.1)

## player should have played atleast 45 minutes ##
playmakerDataFinalFiltered = playmakerDataFinal[playmakerDataFinal["minsPlayed"] > 45]

playmakerDataFinalFiltered.loc[
    playmakerDataFinalFiltered["playmakerRating"].nlargest(10).index]\
        [["player.name", "team.name", "playmakerRating"]]

## Messi leads the top 10 list ##
