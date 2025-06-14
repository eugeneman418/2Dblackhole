import numpy as np
from scipy.integrate import solve_ivp
class Schwarzchild2D:
    def __init__(self, M):
        self.M = M
        self.r_S = 2 * M  # Schwarzschild radius

    def get_dynamics(self, initial_conditions, batched=True):

        if not batched:
            r0 = initial_conditions[0]
            v_phi0 = initial_conditions[3]
        else:
            r0 = initial_conditions[:, 0]
            v_phi0 = initial_conditions[:, 3]

        angular_momentum = r0**2 * v_phi0

        if not batched:
            y0 = initial_conditions[:3] # v_phi is baked into angular momentum, which is constant
        else:
            y0 = initial_conditions[:, :3].flatten()  # Flatten to a 1D array for the initial state
            batch_size = initial_conditions.shape[0]

        def f(t,y):
            if not batched:
                r, phi, v_r = y
            else:
                y = y.reshape((batch_size, 3))
                r, phi, v_r = y[:, 0], y[:, 1], y[:, 2]

            dr, dphi, dv_r = v_r, angular_momentum / r**2, angular_momentum**2 * (r - 3*self.M) / r**4

            if not batched:
                dy = np.array([dr, dphi, dv_r])
            else:
                dy = np.empty_like(y)
                dy[:, 0] = dr
                dy[:, 1] = dphi
                dy[:, 2] = dv_r

            if not batched:
                if r <= self.r_S:
                    dy = np.zeros_like(dy)
            else:
                dy[r <= self.r_S] = 0

            return dy.flatten()
        
        return f, y0    
        
    
    def simulate(self, initial_conditions, start_time, end_time, batched=True, rtol=1e-12, atol=1e-15):
        f, y0 = self.get_dynamics(initial_conditions, batched=batched)
        t_span = (start_time, end_time)
        sol = solve_ivp(f, t_span, y0, vectorized=batched, rtol=rtol, atol=atol)
        if not batched:
            return sol.t, sol.y.T
        else:
            num_steps = sol.t.shape[0]
            batch_size = initial_conditions.shape[0]
            print("batched output shape", sol.y.shape)
            y = sol.y.reshape(batch_size, 3, num_steps)
            return sol.t, np.transpose(y, axes=(0, 2, 1))  # batch_size, num_steps, 3
    
    def get_asymptotic_direction(self, trajectory, batched=True):
        if not batched:
            r = trajectory[-1, 0]
            phi = trajectory[-1, 1]
            if r <= self.r_S:
                return np.nan
            else:
                return phi
        else:
            phi = trajectory[:,-1, 1]
            r = trajectory[:,-1, 0]
            return np.where(r <= self.r_S, np.nan, phi)


def cartesian_to_polar(cartesian_coords, batched=True):
    """
    converts cartesian coordinates to polar coordinates
    """
    if not batched:
        cartesian_coords = np.expand_dims(cartesian_coords, axis=0)

    x = cartesian_coords[:, 0]
    y = cartesian_coords[:, 1]
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    polar_coords = np.stack((r, phi), axis=-1)
    return polar_coords if batched else polar_coords[0]

def polar_to_cartesian(polar_coords, batched=True):
    """
    converts polar coordinates to cartesian coordinates
    """
    if not batched:
        polar_coords = np.expand_dims(polar_coords, axis=0)

    r = polar_coords[:, 0]
    phi = polar_coords[:, 1]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    cartesian_coords = np.stack((x, y), axis=-1)
    return cartesian_coords if batched else cartesian_coords[0]

def polar_jacobian(cartesian_coords, batched=True):
    """
    computes jacobian of polar coordinates wrt. cartesian
    """
    polar_coords = cartesian_to_polar(cartesian_coords, batched=batched)
    if not batched:
        polar_coords = np.expand_dims(polar_coords, axis=0)

    r = polar_coords[:, 0]
    phi = polar_coords[:, 1]
    jac = np.zeros((cartesian_coords.shape[0], 2, 2))
    jac[:, 0, 0] = np.cos(phi)
    jac[:, 0, 1] = np.sin(phi)
    jac[:, 1, 0] = -np.sin(phi) / r
    jac[:, 1, 1] = np.cos(phi) / r
    return jac if batched else jac[0]

def cartesian_jacobian(polar_coords, batched=True):
    """
    computes jacobian of cartesian coordinates wrt. polar
    """
    if not batched:
        polar_coords = np.expand_dims(polar_coords, axis=0)

    r = polar_coords[:, 0]
    phi = polar_coords[:, 1]
    jac = np.zeros((polar_coords.shape[0], 2, 2))
    jac[:, 0, 0] = np.cos(phi)
    jac[:, 0, 1] = -r * np.sin(phi)
    jac[:, 1, 0] = np.sin(phi)
    jac[:, 1, 1] = r * np.cos(phi)
    return jac if batched else jac[0]

if __name__ == "__main__":
    # Example usage
    schwarzschild = Schwarzchild2D(M=1)

    initial_conditions = np.array([
        [3.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0]
        ])  # r, phi, v_r, v_phi
    t, y = schwarzschild.simulate(initial_conditions, start_time=0, end_time=10, batched=True)
    print("batch", y.shape)

    for i in range(3,6):
        initial_conditions = np.array(
            [3.0, 0.0, 0.0, 1.0],
            )  # r, phi, v_r, v_phi
        t, y = schwarzschild.simulate(initial_conditions, start_time=0, end_time=10, batched=False)
        print(f"photon {i}", y.shape)

