import numpy as np

import constants
from util import seperate_df

class Fields:
  def __init__(self, base_fields):
    self.base   = base_fields
    self.foravg = [f'For{f}Avg' for f in base_fields]
    self.oppavg = [f'Opp{f}Avg' for f in base_fields]
    self.Ta_avg = [f'Ta{f}' for f in self.foravg] + [f'Ta{f}' for f in self.oppavg]
    self.Tb_avg = [f'Tb{f}' for f in self.foravg] + [f'Tb{f}' for f in self.oppavg]
    self.win    = [f'W{f}' for f in base_fields]
    self.loss   = [f'L{f}' for f in base_fields]
  
class Predictor:
  def __init__(self, stats_df, input_fields, output_fields, train_ratio=constants.train_ratio):
    self.input_fields = input_fields
    self.output_fields = output_fields
    self.train_ratio = train_ratio
    self.train_df, self.test_df = seperate_df(stats_df, train_ratio=train_ratio)
    self.model = None
  
  def run(self):
    pass
  def results(self):
    pass
  def train(self):
    pass

class LeastSquares(Predictor):
  def run(self, df):
    input_stats = df[self.input_fields].to_numpy()
    return input_stats.dot(self.model)
  
  def results(self, df):
    pred_stats  = self.run(df)
    true_stats  = df[self.output_fields].to_numpy()
    return pred_stats, true_stats

  def train(self):
    input_stats  = self.train_df[self.input_fields].to_numpy()
    output_stats = self.train_df[self.output_fields].to_numpy()
    self.model, _, _, _ = np.linalg.lstsq(input_stats, output_stats, rcond=None)
    return self.results(self.train_df)