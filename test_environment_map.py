import numpy as np
import scipy as sp
import unittest
from environment_map import EnvironmentMapPolar

class TestEnvironmentMap(unittest.TestCase):
    def setUp(self):
        self.N = 1000
        self.data = np.random.rand(self.N)
        self.env_map = EnvironmentMapPolar(self.data)
        self.theta_vals = np.linspace(0, 2 * np.pi, self.N+1, endpoint=True)
        interpolate_data = list(self.data)
        interpolate_data.append(self.data[0]) # scipy interpolator doesn't wrap around
        self.interpolator = sp.interpolate.interp1d(
            self.theta_vals, np.array(interpolate_data), kind='linear', assume_sorted=True
        )

    def test_get_value(self):
        for i in range(self.N+1):
            theta = (2*np.pi / self.N) * i
            expected = self.data[ i % self.N]
            actual = self.env_map.lookup(theta)
            self.assertAlmostEqual(actual, expected)
    
    def test_get_value_batch(self):
        theta = np.arange(self.N+1) * 2*np.pi / self.N
        expected = self.data[np.arange(self.N+1) % self.N]
        actual = self.env_map.lookup(theta)

        almost_equal = np.allclose(actual, expected)
        self.assertTrue(almost_equal)

    def test_average(self):
        expected = np.mean(self.data)
        actual = self.env_map.integrate(0, 2*np.pi, batched=False)
        self.assertAlmostEqual(actual, expected, places=3)

    def test_average_batch(self):
        batch_size = 100
        expected = np.mean(self.data) * np.ones(batch_size)
        actual = self.env_map.integrate(np.zeros(batch_size), 2*np.pi * np.ones(batch_size), batched=True)
        
        for i in range(batch_size):
            self.assertAlmostEqual(expected[i], actual[i], places=3)

        for i in range(batch_size-1):
            self.assertAlmostEqual(actual[i], actual[i+1], places=3)

    
    def test_average_wrap_batch(self):
        batch_size = 100
        expected = np.mean(self.data) * np.ones(batch_size)
        actual = self.env_map.integrate(np.ones(batch_size), (1-1e-10)* np.ones(batch_size), batched=True)
        
        for i in range(batch_size):
            self.assertAlmostEqual(expected[i], actual[i], places=3)

        for i in range(batch_size-1):
            self.assertAlmostEqual(actual[i], actual[i+1], places=3)
    
    def test_average_zero_batch(self):
        batch_size = 100
        expected = np.zeros(batch_size)
        actual = self.env_map.integrate(np.ones(batch_size), np.ones(batch_size), batched=True)
        
        for i in range(batch_size):
            self.assertAlmostEqual(expected[i], actual[i], places=3)

        for i in range(batch_size-1):
            self.assertAlmostEqual(actual[i], actual[i+1], places=3)
            
if __name__ == '__main__':
    unittest.main()