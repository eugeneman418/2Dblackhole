import numpy as np
import scipy as sp

class EnvironmentMapPolar:
    def __init__(self, data):
        self.N = len(data)
        self.data = np.empty(len(data)+1)
        self.angles = np.linspace(0, 2*np.pi, num=len(data)+1, endpoint=True) # 0 to 2pi in increments of 2pi/len(data)
        self.data[:-1] = data
        self.data[-1] = data[0] # for wrapping around
        self.interpolator = sp.interpolate.interp1d(
            self.angles, self.data, kind='linear', assume_sorted=True
        )

    def lookup(self, theta):
        return self.interpolator(theta % (2 * np.pi) )
    
    def integrate(self, theta_start, theta_end, average = True, batched=True):
        # if theta_start > theta_end then integrate from start to 2pi then 0 to theta end
        if not batched:
            theta_start = np.expand_dims(theta_start, axis=0)
            theta_end = np.expand_dims(theta_end, axis=0)
        
        results = np.zeros(len(theta_start))
        
        wrap_mask = theta_start > theta_end
        if np.sum(wrap_mask) > 0: # contains True
            upper = 2*np.pi * np.ceil(theta_start[wrap_mask] / (2*np.pi))
            lower = 2*np.pi * np.floor(theta_end[wrap_mask] / (2*np.pi))
            integral = self.integrate(theta_start[wrap_mask], upper, average=False, batched=True) + self.integrate(lower, theta_end[wrap_mask], average=False, batched=True)
            interval = upper - theta_start[wrap_mask] + theta_end[wrap_mask] - lower
            nonzero = interval != 0
            results[wrap_mask & nonzero] = integral[nonzero] if not average else integral[nonzero] / interval[nonzero]

        unwrap_mask = theta_start < theta_end
        if np.sum(unwrap_mask) > 0:
            phi = np.linspace(theta_start[unwrap_mask], theta_end[unwrap_mask], num=self.N).T
            values = self.interpolator(phi)
            integral = np.trapezoid(values, phi, axis=1)
            results[unwrap_mask] = integral if not average else integral / (theta_end[unwrap_mask] - theta_start[unwrap_mask])

        return results if batched else results[0]
