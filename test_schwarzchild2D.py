from schwarzchild2D import  *
import numpy as np
import unittest

class TestSchwarzchild(unittest.TestCase):

    def test_polar_jacobian(self):
        cartesian = np.random.rand(10,2)
        identities = np.tile(np.eye(2), (10, 1, 1))
        polar =  cartesian_to_polar(cartesian, batched=True)
        J = polar_jacobian(cartesian, batched=True)
        Jinv = cartesian_jacobian(polar, batched=True)
        product = np.einsum('ijk,ikl->ijl', J, Jinv)
        self.assertTrue(np.allclose(identities, product))
    
    def test_cartesian_jacobian(self):
        polar = np.random.rand(10,2)
        identities = np.tile(np.eye(2), (10, 1, 1))
        cartesian = polar_to_cartesian(polar, batched=True)
        J = cartesian_jacobian(polar, batched=True)
        Jinv = polar_jacobian(cartesian, batched=True)
        product = np.einsum('ijk,ikl->ijl', J, Jinv)
        self.assertTrue(np.allclose(identities, product))

    def test_boundary_direction(self):
        blackhole = Schwarzchild2D(1)
        trajectory = np.array([
            [0,1,2,3], # r 
            [4,5,6,7], # phi
            [0,0,0,0], # v_r
        ]).T
        boundary_direction = blackhole.get_boundary_direction(trajectory, boundary_radius=2.5, batched=False, interpolate=False)
        self.assertEqual(6, boundary_direction)
        
    def test_boundary_direction_event_horizon(self):
        blackhole = Schwarzchild2D(1)
        trajectory = np.array([
            [0,1,2,3], # r 
            [4,5,6,7], # phi
            [0,0,0,0], # v_r
        ]).T
        boundary_direction = blackhole.get_boundary_direction(trajectory, boundary_radius=5, batched=False, interpolate=False)
        self.assertTrue(np.isnan(boundary_direction))

    def test_boundary_direction_beyond_bound(self):
        blackhole = Schwarzchild2D(1)
        trajectory = np.array([
            [0.5,1,2,3], # r 
            [4,5,6,7], # phi
            [0,0,0,0], # v_r
        ]).T
        boundary_direction = blackhole.get_boundary_direction(trajectory, boundary_radius=0.1, batched=False, interpolate=False)
        self.assertTrue(np.isnan(boundary_direction))
    
    def test_boundary_direction_interpolate1(self):
        blackhole = Schwarzchild2D(1)
        X = np.array([
            [0,0.5],
            [1,0.5],
        ])
        polar = cartesian_to_polar(X)

        trajectory = np.zeros((2,3))
        trajectory[:,0:2] = polar


        boundary_direction = blackhole.get_boundary_direction(trajectory, boundary_radius=1, batched=False, interpolate=True)
        expected = cartesian_to_polar(np.array([0.86603, 0.5]), batched=False)[1]
        
        self.assertAlmostEqual(expected, boundary_direction, places=4)

    def test_boundary_direction_interpolate2(self):
        blackhole = Schwarzchild2D(1)
        X = np.array([
            [0,0],
            [1,1],
        ])
        polar = cartesian_to_polar(X)

        trajectory = np.zeros((2,3))
        trajectory[:,0:2] = polar


        boundary_direction = blackhole.get_boundary_direction(trajectory, boundary_radius=1, batched=False, interpolate=True)
        expected = np.pi/4
        
        self.assertAlmostEqual(expected, boundary_direction, places=4)
        
    def test_line_circle_intersection(self):  
        start = np.array([
            [2,-2], # no intersection
            [0,0], # 1 intersection
            [0,-2], # 2 intersections
            [1,-2], # tangent
        ])

        end = np.array([
            [2,2],
            [0,2],
            [0,2],
            [1,2],
        ])
        
        r = 1.0
        result = line_circle_intersection(start, end, r)
        expected = np.array([
            [np.nan, np.nan],
            [0,1],
            [0, -1],
            [1,0],
        ])
        for i in range(len(expected)):
            if np.isnan(expected[i]).any():
                self.assertTrue(np.isnan(result[i]).all())
            else:
                np.testing.assert_allclose(result[i], expected[i], rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
