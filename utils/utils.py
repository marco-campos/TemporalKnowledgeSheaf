import copy

class EarlyStopMonitor(object):
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance
    self.best_state = None

  def early_stop_check(self, curr_val, model):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
      self.best_state = copy.deepcopy(model.state_dict())
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count

      self.best_state = copy.deepcopy(model.state_dict())
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round