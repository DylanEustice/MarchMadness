import numpy as np

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
  def __init__(self, input_fields, output_fields):
    self.input_fields = input_fields
    self.output_fields = output_fields
    self.model = None
  
  def train(self):
    return None
  
  def predict(self):
    return None

class LeastSquares(Predictor):
  def train(self, df):
    A = df[self.input_fields]
    b = df[self.output_fields]
    self.model = np.linalg.lstsq(A, b)