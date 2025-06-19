import numpy as np
import heapq
import schwarzchild2D
from cell import Cell
from environment_map import EnvironmentMapPolar

class UniformGrid(EnvironmentMapPolar):
    def __init__(self, blackhole, camera_cartesian, boundary_radius, num_grid_points, initial_velocity=1, start_time=0, end_time=1e10):
        self.num_grid_points = num_grid_points
        self.grid_phi = np.linspace(0, 2*np.pi, num=self.num_grid_points, endpoint=False) # phi of grid points

        camera_polar = schwarzchild2D.cartesian_to_polar(camera_cartesian, batched=False)
        
        v0_cartesian = np.empty((2, self.num_grid_points)) # initial velocity in cartesian coordinates
        v0_cartesian[0] = np.cos(self.grid_phi) * initial_velocity
        v0_cartesian[1] = np.sin(self.grid_phi) * initial_velocity
        v0_polar = (schwarzchild2D.polar_jacobian(camera_cartesian, batched=False) @ v0_cartesian)

        initial_conditions = np.empty((self.num_grid_points, 4))
        initial_conditions[:,0] = camera_polar[0] # r0
        initial_conditions[:,1] = camera_polar[1] # r1
        initial_conditions[:,2] = v0_polar[0] # v_r0
        initial_conditions[:,3] = v0_polar[1] # v_phi0

        _, self.trajectories = blackhole.simulate(initial_conditions, start_time, end_time)

        self.boundary_indices = schwarzchild2D.get_boundary_intersection(self.trajectories, boundary_radius)

        trajectories_phi = self.trajectories[:,:,1]

        env_phi = np.full_like(self.boundary_indices, np.nan)
        is_valid = ~np.isnan(self.boundary_indices)
        env_phi[is_valid] = trajectories_phi[np.arange(len(self.boundary_indices))[is_valid], self.boundary_indices[is_valid].astype(int)]
        
        super().__init__(env_phi)


class AdaptiveGrid:
    def __init__(self, blackhole, camera_cartesian, boundary_radius, num_grid_points, initial_velocity=1, start_time=0, end_time=1e10):
        self.max_grid_points = num_grid_points
        self.blackhole = blackhole
        self.camera_cartesian = camera_cartesian
        self.camera_polar = schwarzchild2D.cartesian_to_polar(camera_cartesian, batched=False)
        self.boundary_radius = boundary_radius
        self.initial_velocity = initial_velocity
        self.start_time = start_time
        self.end_time = end_time
        
        leaves = [Cell()]

    def compute_grid_point(self, phi0):
        v0_cartesian = np.array([np.cos(phi0), np.sin(phi0)]) * self.initial_velocity        
        v0_polar = (schwarzchild2D.polar_jacobian(self.camera_cartesian, batched=False) @ v0_cartesian)
        initial_condition = np.array([
            self.camera_polar[0],
            self.camera_polar[1],
            v0_polar[0],
            v0_polar[1]
        ])

        _, trajectory = blackhole.simulate(initial_condition, self.start_time, self.end_time, batched=False)

        boundary_idx = schwarzchild2D.get_boundary_intersection(trajectory, self.boundary_radius, batched=False)

        trajectories_phi = trajectory[:,1]
        
        if np.isnan(boundary_idx):
            env_phi = np.nan
        else:
            env_phi = trajectories_phi[boundary_idx]
            trajectory[:boundary_idx+1]

        env_phi = np.full_like(self.boundary_indices, np.nan)
        is_valid = ~np.isnan(self.boundary_indices)
        env_phi[is_valid] = trajectories_phi[np.arange(len(self.boundary_indices))[is_valid], self.boundary_indices[is_valid].astype(int)]
        
        super().__init__(env_phi)
        

        initial_conditions = np.empty((self.num_grid_points, 4))
        initial_conditions[:,0] = self.camera_polar[0] # r0
        initial_conditions[:,1] = camera_polar[1] # r1
        initial_conditions[:,2] = v0_polar[0] # v_r0
        initial_conditions[:,3] = v0_polar[1] # v_phi0



    
    
if __name__ == "__main__":
    blackhole = schwarzchild2D.Schwarzchild2D(1)
    uniform = UniformGrid(blackhole, np.array([-3,0]), 5, num_grid_points=100)
    print(uniform.boundary_indices.shape)
    print(uniform.data.shape)
