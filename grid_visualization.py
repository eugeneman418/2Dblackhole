import pygame
import pygame.gfxdraw
import sys
import numpy as np
import utils
import schwarzchild2D
from environment_map import EnvironmentMapPolar
from grid import UniformGrid


pygame.init()

WIDTH, HEIGHT = 1280, 720
DISTANCE_SCALE = 0.1
VELOCITY_SCALE = 1e-3
BLACKHOLE_CENTER = (WIDTH // 2, HEIGHT // 2)

START_TIME, END_TIME = 0, 2e2
ENVIRONMENT_MAP_THICKNESS = 10
ENVIRONMENT_MAP_RADIUS = min(WIDTH, HEIGHT)//2 - ENVIRONMENT_MAP_THICKNESS
ENVIRONMENT_MAP = EnvironmentMapPolar(np.linspace(0,1,num=10000, endpoint=False))
ENVIRONMENT_MAP_RESOLUTION = 500

CAMERA_POSITION = (WIDTH // 2 - ENVIRONMENT_MAP_RADIUS * 0.6 , HEIGHT // 2)
CAMERA_RADIUS = ENVIRONMENT_MAP_RADIUS // 8
CAMERA_THICKNESS = ENVIRONMENT_MAP_THICKNESS * 3
CAMERA_RESOLUTION = 50


blackhole = schwarzchild2D.Schwarzchild2D(1)
blackhole_radius_pixels = int(blackhole.r_S / DISTANCE_SCALE)
camera_cartesian = utils.screen_to_blackhole_coords(CAMERA_POSITION, WIDTH, HEIGHT, DISTANCE_SCALE)

uniform_grid = UniformGrid(blackhole, camera_cartesian, ENVIRONMENT_MAP_RADIUS * DISTANCE_SCALE, num_grid_points=100, start_time=START_TIME, end_time=END_TIME)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Blackhole Grid Visualization")

WHITE = (255, 255, 255)
GREY = (127,127,127)
BLACK = (0, 0, 0)

screen.fill(WHITE)

draw_grid = True


clock = pygame.time.Clock()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


    screen.fill(WHITE)
    utils.draw_halo(screen, BLACKHOLE_CENTER, ENVIRONMENT_MAP_RADIUS, ENVIRONMENT_MAP_THICKNESS, ENVIRONMENT_MAP.lookup, resolution=ENVIRONMENT_MAP_RESOLUTION)
    pygame.gfxdraw.aacircle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)
    pygame.gfxdraw.filled_circle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)

    for i, trajectory_polar in enumerate(uniform_grid.trajectories):
        trajectory_cartesian = schwarzchild2D.polar_to_cartesian(trajectory_polar, batched=True)
        boundary_idx = uniform_grid.boundary_indices[i]
        if ~np.isnan(boundary_idx):
            boundary_idx = int(boundary_idx)
            trajectory_cartesian = trajectory_cartesian[:boundary_idx+1]
        trajectory_screen = [utils.blackhole_coords_to_screen_coords(pos, WIDTH, HEIGHT, DISTANCE_SCALE) for pos in trajectory_cartesian]
        pygame.draw.aalines(screen, GREY, False, trajectory_screen)

    utils.draw_halo(screen, CAMERA_POSITION, CAMERA_RADIUS, CAMERA_THICKNESS, lambda phi: ENVIRONMENT_MAP.lookup(uniform_grid.lookup(phi)), resolution=CAMERA_RESOLUTION, antialias=100)
    

    pygame.display.flip()
    clock.tick(30)



