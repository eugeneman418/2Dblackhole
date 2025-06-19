import numpy as np
import math
import pygame
import pygame.gfxdraw
import colorsys

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


def draw_arrow(surface, start, end, color=(0, 0, 0), arrow_size=10):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)

    x1 = end[0] - arrow_size * math.cos(angle - math.pi / 6)
    y1 = end[1] - arrow_size * math.sin(angle - math.pi / 6)
    x2 = end[0] - arrow_size * math.cos(angle + math.pi / 6)
    y2 = end[1] - arrow_size * math.sin(angle + math.pi / 6)

    pygame.draw.aaline(surface, color, start, end)

    points = [(end[0], end[1]), (x1, y1), (x2, y2)]
    pygame.gfxdraw.filled_polygon(surface, points, color)
    pygame.gfxdraw.aapolygon(surface, points, color)

def draw_environment_map(surface, center, radius, thickness, environment_map, resolution=360):
    angles, step_size = np.linspace(0, 2*np.pi, num=resolution, endpoint=False, retstep=True)

    inner_radius = radius
    outer_radius = radius + thickness

    for angle in angles:
        hue = environment_map(angle)
        rgb = colorsys.hsv_to_rgb(hue, 1, 1)
        color = tuple(int(c * 255) for c in rgb)

        start_angle = angle
        stop_angle = angle + step_size

        x1_inner = center[0] + inner_radius * math.cos(start_angle)
        y1_inner = center[1] + inner_radius * math.sin(start_angle)
        x1_outer = center[0] + outer_radius * math.cos(start_angle)
        y1_outer = center[1] + outer_radius * math.sin(start_angle)

        x2_inner = center[0] + inner_radius * math.cos(stop_angle)
        y2_inner = center[1] + inner_radius * math.sin(stop_angle)
        x2_outer = center[0] + outer_radius * math.cos(stop_angle)
        y2_outer = center[1] + outer_radius * math.sin(stop_angle)

        points = [(x1_inner, y1_inner), (x1_outer, y1_outer), (x2_outer, y2_outer), (x2_inner, y2_inner)]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)


def line_circle_intersection(start, end, r):
    """
    start & end are Nx2 batch of points
    circle is centered at origin with radius r
    returns first intersection if there are many
    returns nan if no intersections
    """

    dir = end - start

    a = np.sum(dir**2, axis=1)
    b = 2 * np.sum(start * dir, axis=1)
    c = np.sum(start**2, axis=1) - r**2

    discriminant = b**2 - 4 * a * c

    result = np.full_like(start, np.nan, dtype=np.float64)

    valid = discriminant >= 0
    sqrt_disc = np.sqrt(discriminant[valid])

    a_valid = a[valid]
    b_valid = b[valid]
    dir_valid = dir[valid]
    start_valid = start[valid]

    t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
    t2 = (-b_valid + sqrt_disc) / (2 * a_valid)

    t = np.where((0 <= t1) & (t1 <= 1), t1,
         np.where((0 <= t2) & (t2 <= 1), t2, np.nan))
    
    has_intersection = ~np.isnan(t)
    idxs = np.nonzero(valid)[0][has_intersection]

    result[idxs] = start_valid[has_intersection] + t[has_intersection, np.newaxis] * dir_valid[has_intersection]


    return result
