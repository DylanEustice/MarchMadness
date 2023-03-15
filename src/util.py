import constants
import pandas as pd
import numpy as np
import os

def find_team_games(df, seas, team):
  is_seas = seas == df['Season']
  is_team = (df['WTeamID'] == team) | (df['LTeamID'] == team)
  return df[is_seas & is_team]

'''
Load in a CSV data file, dropping any seasons before 'first_season' and any
corresponding to seasons in 'drop_seasons'
'''
def load_csv(
  filename,
  first_season=constants.first_season,
  drop_seasons=constants.drop_seasons,
  data_path=constants.data_path
):
  df = pd.read_csv(os.path.join(data_path, filename))
  if 'Season' in df.columns:
    df.drop(df[df['Season'] < first_season].index, inplace=True)
    for seas in drop_seasons:
      df.drop(df[df['Season'] == seas].index, inplace=True)
  return df.reset_index(drop=True)

def save_csv(df, filename, data_path=constants.data_path):
  df.to_csv(os.path.join(data_path, filename), index=False)

'''
Load teams, adding conference ID
'''
def load_teams(
  first_season=constants.first_season,
  drop_seasons=constants.drop_seasons,
  data_path=constants.data_path
):
  # Make a conference dictionary
  conf_df = load_csv(
    'Conferences.csv',
    first_season=first_season,
    drop_seasons=drop_seasons,
    data_path=data_path
  )
  conf_df['ConfID'] = conf_df.index
  conf_dict = dict(zip(conf_df.ConfAbbrev, conf_df.ConfID))
  # Add conference ID's and return
  team_df = load_csv(
    'MTeamConferences.csv',
    first_season=first_season,
    drop_seasons=drop_seasons,
    data_path=data_path
  )
  team_df['ConfId'] = list(map(conf_dict.get, team_df['ConfAbbrev']))
  return team_df

'''
Return a test and train dataset
'''
def seperate_df(df, train_ratio=constants.train_ratio):
  ix_train = np.random.random(df.shape[0]) < train_ratio
  train_df = df.loc[df.index[ix_train]]
  test_df  = df.loc[df.index[~ix_train]]
  return train_df, test_df

'''
Return the percentage of the time the predicted winner matched the real winner
'''
def winrate(pred, real):
  pred_winner = pred[:,0] > pred[:,1]
  real_winner = real[:,0] > real[:,1]
  return (pred_winner == real_winner).mean()

'''
Compute the mean square error
'''
def mse(a, b, axis=0):
  return np.sqrt(((a - b)**2).mean(axis=axis))