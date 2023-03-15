import util
import argparse
import numpy as np

import constants
from objects import Fields, LeastSquares

if __name__ == "__main__":
  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--first_season', default=constants.first_season)
  parser.add_argument('--drop_seasons', default=constants.drop_seasons)
  args = vars(parser.parse_args())

  # Load data
  teams_df = util.load_csv(
    'MTeamConferencesWithAvg.csv',
    first_season=args['first_season'],
    drop_seasons=args['drop_seasons']
  )
  stats_df = util.load_csv(
    'MRegularSeasonDetailedResultsWithAvg.csv',
    first_season=args['first_season'],
    drop_seasons=args['drop_seasons']
  )

  # Build the Least-Squares model
  input_fields_base = Fields(['Score', 'FGM', 'FGA', 'FGM3', 'FGA3'])
  input_fields = input_fields_base.Ta_avg + input_fields_base.Tb_avg
  output_fields = ['TaScore', 'TbScore']

  lspred = LeastSquares(stats_df, teams_df, input_fields, output_fields)
  pred_train, true_train = lspred.train()
  pred_test, true_test   = lspred.results(lspred.test_df)

  # Compute results and print
  train_wr  = util.winrate(pred_train, true_train)
  train_mse = util.mse(np.diff(pred_train, axis=1), np.diff(true_train, axis=1))
  test_wr   = util.winrate(pred_test, true_test)
  test_mse  = util.mse(np.diff(pred_test, axis=1), np.diff(true_test, axis=1))

  print(f'Train winrate, err: {100*train_wr:0.4}, {train_mse}')
  print(f'Test winrate, err: {100*test_wr:0.4}, {test_mse}')