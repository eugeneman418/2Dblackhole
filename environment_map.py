import numpy as np

class EnvironmentMapPolar:
    def __init__(self, data):
        self.data = data # if len(data) = N then data[i] is value at 2pi/N * i
        self.step_size = 2 * np.pi / len(data)
        self.integral_table = 0.5 * (data[1:] + data[:-1]) * self.step_size # integral_table[i] = integral from 2pi/N * i to 2pi/N * (i+1)
        self.sat = np.cumsum(self.integral_table)  # sat[i] = integral from 0 to 2pi/N * (i+1)

    def get_value(self, theta):
        theta = theta % (2 * np.pi) 
        N = len(self.data)
        idx = theta * N / (2 * np.pi)
        lower = np.floor(idx).astype(int)
        upper = np.ceil(idx).astype(int)

        lower_value = self.data[lower % N]
        upper_value = self.data[upper % N]

        # linear interpolation
        lower_theta = lower * 2 * np.pi / N
        upper_theta = upper * 2 * np.pi / N
        
        if lower == upper:
            return lower_value
        else:
            return lower_value * (upper_theta - theta) / (upper_theta - lower_theta) + upper_value * (theta - lower_theta) / (upper_theta - lower_theta)
    
    def integrate(self, theta_start, theta_end):
        # if theta_start > theta_end then integrate from start to 2pi then 0 to theta end

        theta_start = theta_start % (2 * np.pi)
        theta_end = theta_end % (2 * np.pi)
        
        if theta_start > theta_end:
            theta_start, theta_end = theta_end, theta_start

        if theta_start == theta_end:
            return 0

        a_idx = np.ceil(theta_start * len(self.data) / (2 * np.pi)).astype(int)
        b_idx = np.floor(theta_end * len(self.data) / (2 * np.pi)).astype(int)

        a = 2*np.pi / len(self.data) * a_idx
        b = 2*np.pi / len(self.data) * b_idx

        #split integral into 3 parts, theta_start to a, a to b, b to theta_end
        integral = 0
        if a != theta_start:
            integral += 0.5 * (self.get_value(theta_start) + self.data[a_idx]) * (a - theta_start)
        if a_idx < b_idx:
            integral += self.sat[b_idx-1] - self.sat[a_idx-1]
        if b != theta_end:
            integral += 0.5 * (self.data[b_idx] + self.get_value(theta_end)) * (theta_end - b)
    
        return integral

if __name__ == "__main__":
    # Example usage
    data = np.array([0, 2, 4, 6, 8])
    env_map = EnvironmentMapPolar(data)

    theta = 2 * np.pi / 5 * 0.1
    value = env_map.get_value(theta)
    print(f"Value at theta={theta}: {value}")

    theta = 2 * np.pi / 5 * 3.9
    value = env_map.get_value(theta)
    print(f"Value at theta={theta}: {value}")

    theta_start = 2 * np.pi / 5 * 0.1
    theta_end = 2 * np.pi / 5 * 3.9
    integral = env_map.integrate(theta_start, theta_end)
    print(f"Integral from {theta_start} to {theta_end}: {integral}")