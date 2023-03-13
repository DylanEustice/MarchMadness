import constants
import pandas as pd
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

'''
Load teams, adding conference ID
'''
def load_teams(data_path=constants.data_path):
  # Make a conference dictionary
  conf_df = load_csv('Conferences.csv', data_path=data_path)
  conf_df['ConfID'] = conf_df.index
  conf_dict = dict(zip(conf_df.ConfAbbrev, conf_df.ConfID))
  # Add conference ID's and return
  team_df = load_csv('MTeamConferences.csv', data_path=data_path)
  team_df['ConfId'] = list(map(conf_dict.get, team_df['ConfAbbrev']))
  return team_df