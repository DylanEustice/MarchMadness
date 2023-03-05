import constants as const
import pandas as pd

def find_team_games(df, seas, team):
  is_seas = seas == df['Season']
  is_team = (df['WTeamID'] == team) | (df['LTeamID'] == team)
  return df[is_seas & is_team]

def load_csv(filename):
  df = pd.read_csv(filename)
  if 'Season' in df.columns:
    df.drop(df[df['Season'] < const.first_season].index, inplace=True)
  return df.reset_index()