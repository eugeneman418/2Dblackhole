import numpy as np


def screen_to_blackhole_coords(pos, width, height, scale):
    """
    convert screen coordinates to simulation coordinates
    in simulation coordinates, the black hole is at the origin
    in screen coordinates, the black hole is at the center
    width, height: dimensions of the screen
    scale: distance which 1 pixel represents in simulation
    """
    # black hole is at the origin, which corresponds to the center of the screen
    x = pos[0] - width // 2
    y = height // 2 - pos[1] 
    return np.array([x * scale, y * scale])  # scale factor: 100 pixels = 1 Schwarzschild unit

def blackhole_coords_to_screen_coords(sim_pos, width, height, scale):
    x = int(sim_pos[0] / scale + width // 2)
    y = int(height // 2 - sim_pos[1] / scale)
    return (x, y)