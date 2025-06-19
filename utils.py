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

def draw_halo(surface, center, radius, thickness, environment_map, resolution=360, antialias = None):
    angles, step_size = np.linspace(0, 2*np.pi, num=resolution, endpoint=False, retstep=True)

    inner_radius = radius
    outer_radius = radius + thickness

    for angle in angles:
        if antialias is not None: # antialias by supersampling
            supersample_angles = np.linspace(angle, angle+step_size, num=antialias)
            rgb = np.zeros(3)
            for supersample_angle in supersample_angles:
                hue = environment_map(supersample_angle)
                if np.isnan(hue):
                    rgb = np.zeros(3)
                    break
                else:
                    rgb += np.array(colorsys.hsv_to_rgb(hue, 1, 1))
            rgb /=  antialias
        else:
            hue = environment_map(angle)
            rgb = colorsys.hsv_to_rgb(hue, 1, 1) if ~np.isnan(hue) else (0,0,0)
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

