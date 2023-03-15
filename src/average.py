import argparse
import pandas as pd
from tqdm import tqdm

import constants
import util
from objects import Fields

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

if __name__ == "__main__":
  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--first_season', default=constants.first_season)
  parser.add_argument('--drop_seasons', default=constants.drop_seasons)
  args = vars(parser.parse_args())

  # Load data
  team_df = util.load_teams(
    first_season=args['first_season'],
    drop_seasons=args['drop_seasons']
  )
  box_df = util.load_csv(
    'MRegularSeasonDetailedResults.csv',
    first_season=args['first_season'],
    drop_seasons=args['drop_seasons']
  )

  # Compute averages for all box score stats
  fields = Fields(constants.box_stats)
  add_averages(fields, team_df, box_df)

  # Write new files
  util.save_csv(team_df, 'MTeamConferencesWithAvg.csv')
  util.save_csv(box_df, 'MRegularSeasonDetailedResultsWithAvg.csv')