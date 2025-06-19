import numpy as np

class Cell:
    def __init__(self, lower_phi, upper_phi, 
                 lower_value, upper_value, 
                 lower_trajectory, upper_trajectory,
                 left_child=None, right_child=None):
        assert lower_phi < upper_phi, f"expect {lower_phi} < {upper_phi}"
        self.lower_phi = lower_phi
        self.upper_phi = upper_phi

        self.lower_value = lower_value
        self.upper_value = upper_value
        
        self.lower_trajectory = lower_trajectory
        self.upper_trajectory = upper_trajectory

        self.left_child = left_child
        self.right_child = right_child


    def in_range(self, phi):
        return self.lower_phi <= phi <= self.upper_phi
    
    def lookup(self, phi):
        assert self.in_range(phi), f"expected {phi} to be in [{self.lower_phi}, {self.upper_phi}]"

        if np.isnan(self.upper_value) or np.isnan(self.lower_value):
            return np.nan
        
        elif phi == self.lower_phi:
            return self.lower_value
        
        elif phi == self.upper_phi:
            return self.upper_value
        
        elif self.left_child is not None and self.left_child.in_range(phi):
            return self.left_child.lookup(phi)
        elif self.right_child is not None and self.right_child.in_range(phi):
            return self.right_child.lookup(phi)
        else:
            return self.interpolate(phi)
    
    def interpolate(self, phi):
        span = self.upper_phi - self.lower_phi
        return self.lower_value * (self.upper_phi - phi) / span + self.upper_value * (phi - self.lower_phi) / span
    
    def num_children(self):
        num = 0
        if self.left_child is not None:
            num += 1
        if self.right_child is not None:
            num += 1
        return num
    
    def priority(self):
        if np.isnan(self.upper_value) or np.isnan(self.lower_value):
            return 2*np.pi / (self.upper_phi - self.lower_phi)
        
        lower_value = self.lower_value % (2*np.pi)
        upper_value = self.upper_value % (2*np.pi)
        return abs(upper_value - lower_value) / (self.upper_phi - self.lower_phi)

    def refine(self, middle_phi, middle_value, middle_trajectory):
        if self.left_child is None and self.right_child is None:
            self.left_child = Cell(self.lower_phi, middle_phi, 
                                   self.lower_value, middle_value, 
                                   self.lower_trajectory, middle_trajectory)
            self.right_child = Cell(middle_phi, self.upper_phi, 
                                    middle_value, self.upper_value, 
                                    middle_trajectory, self.upper_trajectory)
            return [self.left_child, self.right_child]
        else:
            return []
        
    def __lt__(self, other): # lt because heaq implements a min heaq
        return self.priority > other.priority

    def __repr__(self):
        return f"Cell([{self.lower_phi}, {self.upper_phi}], {self.lower_value}, {self.upper_value}) Number of children: {self.num_children()}"

