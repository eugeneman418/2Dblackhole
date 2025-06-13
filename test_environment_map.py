import numpy as np
import scipy as sp
import unittest
from environment_map import EnvironmentMapPolar

class TestEnvironmentMap(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.data = np.random.rand(self.N)
        self.env_map = EnvironmentMapPolar(self.data)
        self.theta_vals = np.linspace(0, 2 * np.pi, self.N+1, endpoint=True)
        interpolate_data = list(self.data)
        interpolate_data.append(self.data[0]) # scipy interpolator doesn't wrap around
        self.interpolator = sp.interpolate.interp1d(
            self.theta_vals, np.array(interpolate_data), kind='linear', assume_sorted=True
        )

    def test_get_value(self):
        for _ in range(1000):
            theta = np.random.uniform(0, 2 * np.pi)
            expected = self.interpolator(theta)
            actual = self.env_map.get_value(theta)
            self.assertAlmostEqual(actual, expected)

    def test_integrate(self):
        for _ in range(1000):
            thetas = np.random.uniform(2*np.pi / self.N * (self.N-1), 2 * np.pi, size=2)
            theta_start = min(thetas)
            theta_end = max(thetas)
            
            num_points = 10000
            ts = np.linspace(theta_start, theta_end, num_points)
            values = self.interpolator(ts)
            expected = np.trapezoid(values, ts)

            actual = self.env_map.integrate(theta_start, theta_end)

            self.assertAlmostEqual(actual, expected, places=4)
            
if __name__ == '__main__':
    unittest.main()