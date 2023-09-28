import numpy as np

class Solution:
  def __init__(self, dimension, lower_bound, upper_bound):
    self.dimension = dimension
    self.lB = lower_bound  # we will use the same bounds for all parameters
    self.uB = upper_bound
    self.parameters = np.zeros(self.d) #solution parameters
    self.f = np.inf  # objective function evaluation