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

        _, trajectories = blackhole.simulate(initial_conditions, start_time, end_time)

        boundary_indices = schwarzchild2D.get_boundary_intersection(trajectories, boundary_radius)

        trajectories_phi = trajectories[:,:,1]

        env_phi = np.full_like(boundary_indices, np.nan)
        is_valid = ~np.isnan(boundary_indices)
        env_phi[is_valid] = trajectories_phi[np.arange(len(boundary_indices))[is_valid], boundary_indices[is_valid].astype(int)]

        self.trajectories = []
        for i, boundary_idx in enumerate(boundary_indices):
            if np.isnan(boundary_idx):
                self.trajectories.append(trajectories[i])
            else:
                self.trajectories.append(trajectories[i][:int(boundary_idx)+1])
        super().__init__(env_phi)


class AdaptiveGrid:
    def __init__(self, blackhole, camera_cartesian, boundary_radius, initial_velocity=1, start_time=0, end_time=1e10):
        self.blackhole = blackhole
        self.camera_cartesian = camera_cartesian
        self.camera_polar = schwarzchild2D.cartesian_to_polar(camera_cartesian, batched=False)
        self.boundary_radius = boundary_radius
        self.initial_velocity = initial_velocity
        self.start_time = start_time
        self.end_time = end_time



    def build_tree(self, max_cell_span, max_depth):
        lower_phi, middle_phi, upper_phi = 0, np.pi, 2*np.pi
        lower_value, lower_trajectory = self.compute_grid_point(lower_phi)
        middle_value, middle_trajectory = self.compute_grid_point(middle_phi)
        upper_value, upper_trajectory = lower_value, lower_trajectory # 0 = 2pi
         
        left_child = Cell(lower_phi, middle_phi, lower_value, middle_value, lower_trajectory, middle_trajectory, depth=1) # 0 to pi
        right_child = Cell(middle_phi, upper_phi, middle_value, upper_value, middle_trajectory, upper_trajectory, depth=1)
        
        self.root = Cell(lower_phi, upper_phi, lower_value, upper_value, lower_trajectory, upper_trajectory, left_child=left_child, right_child=right_child, depth=0)

        refinement_queue = [left_child, right_child]
        heapq.heapify(refinement_queue)

        num_grid_points = 2

        self.leaves = []
        while(len(refinement_queue) > 0):
            cell = heapq.heappop(refinement_queue)
            if cell.span() > max_cell_span and cell.depth < max_depth:
                middle_phi = (cell.lower_phi + cell.upper_phi) / 2
                middle_value, middle_trajectory = self.compute_grid_point(middle_phi)
                for child in cell.refine(middle_phi, middle_value, middle_trajectory):
                    heapq.heappush(refinement_queue, child)

                num_grid_points += 1
            else:
                assert cell.num_children() == 0
                self.leaves.append(cell)

        self.leaves.sort(key=lambda cell: cell.lower_phi)
        return num_grid_points


    def lookup(self,phi):
        return self.root.lookup(phi)


    def compute_grid_point(self, phi0):
        v0_cartesian = np.array([np.cos(phi0), np.sin(phi0)]) * self.initial_velocity        
        v0_polar = (schwarzchild2D.polar_jacobian(self.camera_cartesian, batched=False) @ v0_cartesian)
        initial_condition = np.array([
            self.camera_polar[0],
            self.camera_polar[1],
            v0_polar[0],
            v0_polar[1]
        ])

        _, trajectory = self.blackhole.simulate(initial_condition, self.start_time, self.end_time, batched=False)

        boundary_idx = schwarzchild2D.get_boundary_intersection(trajectory, self.boundary_radius, batched=False)

        trajectories_phi = trajectory[:,1]
        
        if np.isnan(boundary_idx):
            env_phi = np.nan
        else:
            boundary_idx = int(boundary_idx)
            env_phi = trajectories_phi[boundary_idx]
            trajectory = trajectory[:boundary_idx+1] # we only need trajectory up to the boundary for display later

        return env_phi, trajectory



    
    
if __name__ == "__main__":
    blackhole = schwarzchild2D.Schwarzchild2D(1)
    uniform = UniformGrid(blackhole, np.array([-3,0]), 5, num_grid_points=100)
    print(uniform.data.shape)

    adaptive = AdaptiveGrid(blackhole, np.array([-3,0]), 5)
    print(adaptive.build_tree(0.015,5))