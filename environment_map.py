import numpy as np
import scipy as sp

class EnvironmentMapPolar:
    def __init__(self, data):
        self.data = data # if len(data) = N then data[i] is value at 2pi/N * i
        self.step_size = 2 * np.pi / len(data)

    def lookup(self, phi):
        if np.isnan(phi):
            return np.nan
        phi = phi % (2 * np.pi)
        N = len(self.data)
        idx = phi * N / (2 * np.pi)
        lower = np.floor(idx).astype(int)
        upper = np.ceil(idx).astype(int)

        lower_value = self.data[lower % N]
        upper_value = self.data[upper % N]

        # linear interpolation
        lower_phi = lower * 2 * np.pi / N
        upper_phi = upper * 2 * np.pi / N

        if np.isnan(lower_value) or np.isnan(upper_value):
            return np.nan

        elif lower == upper:
            return lower_value
        else:
            return lower_value * (upper_phi - phi) / (upper_phi - lower_phi) + upper_value * (phi - lower_phi) / (upper_phi - lower_phi)