from schwarzchild2D import  cartesian_to_polar, polar_to_cartesian, polar_jacobian, cartesian_jacobian
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
        
if __name__ == '__main__':
    unittest.main()
