import constants
import util

import os, sys
import pandas as pd
import numpy as np

from tqdm import tqdm

datapath = os.path.join('..', 'data', 'march-machine-learning-mania-2023')
base_fields = ['Score', 'FGA', 'FGM', 'FGA3', 'FGM3']

class Fields:
  def __init__(self, base_fields):
    self.base   = base_fields
    self.foravg = [f'For{f}Avg' for f in base_fields]
    self.oppavg = [f'Opp{f}Avg' for f in base_fields]
    self.Ta_avg = [f'Ta{f}' for f in self.foravg] + [f'Ta{f}' for f in self.oppavg]
    self.Tb_avg = [f'Tb{f}' for f in self.foravg] + [f'Tb{f}' for f in self.oppavg]
    self.win    = [f'W{f}' for f in base_fields]
    self.loss   = [f'L{f}' for f in base_fields]

'''
Make a dictionary of conference IDs
'''
def load_conf_dict():
  conf_df = util.load_csv(os.path.join(datapath, 'Conferences.csv'))
  conf_df['ConfID'] = conf_df.index
  return dict(zip(conf_df.ConfAbbrev, conf_df.ConfID))

'''
Load teams, adding conference ID
'''
def load_teams():
  conf_dict = load_conf_dict()
  team_df = util.load_csv(os.path.join(datapath, 'MTeamConferences.csv'))
  team_df['ConfId'] = list(map(conf_dict.get, team_df['ConfAbbrev']))
  return team_df

'''
Given a set of box scores, find the averages stats for a set of either winning
or losing teams
'''
def compute_avg(games, is_winner, fields, new_fields):
  return games[fields.win].where(
    is_winner,
    games[fields.loss].values
  ).mean().rename(
    dict(zip(fields.win, new_fields))
  )

'''
Given a team and the games they played, compute the average set of stats that
the team and its opponents generated
'''
def get_team_averages(games, team_id, fields):
  is_winner = games['WTeamID'] == team_id
  for_avg = compute_avg(games, is_winner, fields, fields.foravg)
  opp_avg = compute_avg(games, ~is_winner, fields, fields.oppavg)
  return for_avg, opp_avg

'''
Add average scores for each team and game
'''
def add_averages(fields, team_df, box_df):
  # Compute averages
  iterable = zip(team_df.index, team_df['Season'], team_df['TeamID'])
  seas_df = None
  for (i, seas, team) in tqdm(iterable, total=team_df.shape[0]):
    # Check if we need to update the set of season games
    if seas_df is None or seas_df.loc[seas_df.index[0], 'Season'] != seas:
      seas_df = box_df[seas == box_df['Season']]

    # Get full season averages for this team
    games = seas_df[(seas_df['WTeamID'] == team) | (seas_df['LTeamID'] == team)]
    for_avg, opp_avg = get_team_averages(games, team, fields)
    team_df.loc[i, fields.foravg] = for_avg
    team_df.loc[i, fields.oppavg] = opp_avg
    team_df.loc[i, 'NumGames']    = games.shape[0]

    # Vectorize compute some stuff
    is_team_a = (team < games[['WTeamID', 'LTeamID']]).any(axis=1)

    # Get averages minus each game
    for (j, day, is_a) in zip(games.index, games['DayNum'], is_team_a):
      game = games.loc[j]
      other_games = games.loc[games['DayNum'] != day]
      for_avg, opp_avg = get_team_averages(other_games, team, fields)

      avg_columns = fields.Ta_avg if is_a else fields.Tb_avg
      averages = pd.concat([for_avg, opp_avg]).rename(
        dict(zip(fields.foravg + fields.oppavg, avg_columns))
      )
      box_df.loc[j, avg_columns] = averages
      box_df.loc[j, 'NumGames'] = games.shape[0] - 1

      score_column = 'TaScore' if is_a else 'TbScore'
      team_score = game['WScore'] if team == game['WTeamID'] else game['LScore']
      box_df.loc[j, score_column] = team_score

'''
Compute the linear least-squares solution for a DataFrame and set of input
and output fields
'''
def compute_lstsq(df, input_fields, output_fields):
  A = df[input_fields]
  b = df[output_fields]
  return np.linalg.lstsq(A, b)

def winrate(df, input_fields, x):
  res = df[input_fields].to_numpy().dot(x)
  pred_winner = res[:,0] > res[:,1]
  real_winner = df['TaScore'] > df['TbScore']
  return (pred_winner == real_winner).mean()

def lsqerr(df, input_fields, output_fields, x):
  res = df[input_fields].to_numpy().dot(x)
  sqr_err = (res - df[output_fields].to_numpy())**2
  return np.sqrt(sqr_err.mean(axis=0))

if __name__ == "__main__":
  fields  = Fields(base_fields)
  team_df = load_teams()
  box_df  = util.load_csv(os.path.join(datapath, 'MRegularSeasonDetailedResults.csv'))
  add_averages(fields, team_df, box_df)

  team_df.to_csv(os.path.join(datapath, 'MTeamConferencesWithAvg.csv'), index=False)
  box_df.to_csv(os.path.join(datapath, 'MRegularSeasonDetailedResultsWithAvg.csv'), index=False)

  ix_train = np.random.random(box_df.shape[0]) < constants.train_ratio
  train_df = box_df.loc[box_df.index[ix_train]]
  test_df  = box_df.loc[box_df.index[~ix_train]]

  input_fields = fields.Ta_avg + fields.Tb_avg
  output_fields = ['TaScore', 'TbScore']
  x, err, _, _ = compute_lstsq(train_df, input_fields, output_fields)
  
  train_wr = winrate(train_df, input_fields, x)
  train_err = lsqerr(train_df, input_fields, output_fields, x)
  print(f'Train winrate, err: {100*train_wr:0.4}, {train_err}')
  
  test_wr = winrate(test_df, input_fields, x)
  test_err = lsqerr(test_df, input_fields, output_fields, x)
  print(f'Test winrate, err: {100*test_wr:0.4}, {test_err}')