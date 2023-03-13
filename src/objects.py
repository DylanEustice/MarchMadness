class Fields:
  def __init__(self, base_fields):
    self.base   = base_fields
    self.foravg = [f'For{f}Avg' for f in base_fields]
    self.oppavg = [f'Opp{f}Avg' for f in base_fields]
    self.Ta_avg = [f'Ta{f}' for f in self.foravg] + [f'Ta{f}' for f in self.oppavg]
    self.Tb_avg = [f'Tb{f}' for f in self.foravg] + [f'Tb{f}' for f in self.oppavg]
    self.win    = [f'W{f}' for f in base_fields]
    self.loss   = [f'L{f}' for f in base_fields]