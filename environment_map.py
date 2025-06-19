import numpy as np
import scipy as sp

class EnvironmentMapPolar:
    def __init__(self, data):
        self.data = data # if len(data) = N then data[i] is value at 2pi/N * i
        self.step_size = 2 * np.pi / len(data)

    def lookup(self, theta):
        theta = theta % (2 * np.pi)
        N = len(self.data)
        idx = theta * N / (2 * np.pi)
        lower = np.floor(idx).astype(int)
        upper = np.ceil(idx).astype(int)

        lower_value = self.data[lower % N]
        upper_value = self.data[upper % N]

        # linear interpolation
        lower_theta = lower * 2 * np.pi / N
        upper_theta = upper * 2 * np.pi / N
        
        if lower == upper:
            return lower_value
        else:
            return lower_value * (upper_theta - theta) / (upper_theta - lower_theta) + upper_value * (theta - lower_theta) / (upper_theta - lower_theta)