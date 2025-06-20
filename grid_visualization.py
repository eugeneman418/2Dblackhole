import pygame
import pygame.gfxdraw
import pygame_gui
import sys
import numpy as np
import utils
import schwarzchild2D
from environment_map import EnvironmentMapPolar
from grid import *


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
CAMERA_RESOLUTION = 500



blackhole = schwarzchild2D.Schwarzchild2D(1)
blackhole_radius_pixels = int(blackhole.r_S / DISTANCE_SCALE)
camera_cartesian = utils.screen_to_blackhole_coords(CAMERA_POSITION, WIDTH, HEIGHT, DISTANCE_SCALE)

max_degree = 70
max_depth = 2
adaptive_grid = AdaptiveGrid(blackhole, camera_cartesian, ENVIRONMENT_MAP_RADIUS * DISTANCE_SCALE, initial_velocity=1)
num_grid_points = adaptive_grid.build_tree(np.deg2rad(max_degree), max_depth)
uniform_grid = UniformGrid(blackhole, camera_cartesian, ENVIRONMENT_MAP_RADIUS * DISTANCE_SCALE, num_grid_points=num_grid_points, start_time=START_TIME, end_time=END_TIME)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Blackhole Grid Visualization")

WHITE = (255, 255, 255)
GREY = (127,127,127)
BLACK = (0, 0, 0)
RED = (127,0,0)
BLUE = (0,0,127)
GREEN = (0,127,0)

screen.fill(WHITE)

draw_uniform = False
draw_adaptive = True
interpolate_with_adaptive = True


clock = pygame.time.Clock()

# ui
degree_slider_scalar = 10 # 1 increment of slider = degree_slider_scalar degrees
manager = pygame_gui.UIManager((WIDTH, HEIGHT))
degree_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((10, 10), (200, 25)),
    start_value=max_degree//degree_slider_scalar,
    value_range=(1, 9),
    manager=manager
)

degree_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((220, 10), (150, 25)),
    text=f"Max Degree: {max_degree:.1f}",
    manager=manager
)

depth_slider = pygame_gui.elements.UIHorizontalSlider(
    relative_rect=pygame.Rect((10, 40), (200, 25)),
    start_value=max_depth,
    value_range=(2, 15),
    manager=manager
)

depth_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((220, 40), (150, 25)),
    text=f"Max Depth: {max_depth}",
    manager=manager
)

grid_point_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((10, 70), (300, 25)),
    text=f"Grid Points: {num_grid_points}",
    manager=manager
)

show_uniform_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 100), (200, 25)),
    text='Show Uniform Grid',
    manager=manager
)

show_adaptive_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 130), (200, 25)),
    text='Hide Adaptive Grid',
    manager=manager
)


interpolation_button = pygame_gui.elements.UIButton(
    relative_rect=pygame.Rect((10, 160), (200, 25)),
    text='Interpolate: Adaptive',
    manager=manager
)

def visualize_uniform_grid():
    for trajectory_polar in uniform_grid.trajectories:
        utils.draw_trajectory(trajectory_polar, screen, GREY, WIDTH, HEIGHT, DISTANCE_SCALE)

def visualize_adaptive_grid():
    for trajectory_polar in [cell.lower_trajectory for cell in adaptive_grid.leaves]:
        utils.draw_trajectory(trajectory_polar, screen, GREEN, WIDTH, HEIGHT, DISTANCE_SCALE)
    
    utils.draw_trajectory(adaptive_grid.leaves[-1].upper_trajectory, screen, GREEN, WIDTH, HEIGHT, DISTANCE_SCALE)

def interpolate_hue(phi, grid):
    value = grid.lookup(phi)
    #print(value)
    return ENVIRONMENT_MAP.lookup(value)


def rebuild_grids():
    global adaptive_grid, uniform_grid, num_grid_points
    adaptive_grid = AdaptiveGrid(blackhole, camera_cartesian, ENVIRONMENT_MAP_RADIUS * DISTANCE_SCALE, initial_velocity=1)
    num_grid_points = adaptive_grid.build_tree(np.deg2rad(max_degree), max_depth)
    uniform_grid = UniformGrid(blackhole, camera_cartesian, ENVIRONMENT_MAP_RADIUS * DISTANCE_SCALE, num_grid_points=num_grid_points, start_time=START_TIME, end_time=END_TIME)

    print("num grid points", num_grid_points, "number of leaves", len(adaptive_grid.leaves))
    for cell in adaptive_grid.leaves:
        print(f"leaf degree {np.rad2deg(cell.span())}, depth {cell.depth}")
        # print(f"lower value {cell.lower_value}, upper value {cell.upper_value}")
    print("----------------------------------")


while True:
    time_delta = clock.tick(30) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        manager.process_events(event)

        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == degree_slider:
                raw_value = degree_slider.get_current_value()
                max_degree = raw_value * degree_slider_scalar
                degree_label.set_text(f"Max Degree: {max_degree:.1f}")
                rebuild_grids()
                grid_point_label.set_text(f"Grid Points: {num_grid_points}")
            elif event.ui_element == depth_slider:
                max_depth = int(depth_slider.get_current_value())
                depth_label.set_text(f"Max Depth: {max_depth}")
                rebuild_grids()
                grid_point_label.set_text(f"Grid Points: {num_grid_points}")

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == show_uniform_button:
                draw_uniform = not draw_uniform
                show_uniform_button.set_text('Hide Uniform Grid' if draw_uniform else 'Show Uniform Grid')
            elif event.ui_element == show_adaptive_button:
                draw_adaptive = not draw_adaptive
                show_adaptive_button.set_text('Hide Adaptive Grid' if draw_adaptive else 'Show Adaptive Grid')

            elif event.ui_element == interpolation_button:
                interpolate_with_adaptive = not interpolate_with_adaptive
                interpolation_button.set_text(f"Interpolate: {'Adaptive' if interpolate_with_adaptive else 'Uniform'}")



    screen.fill(WHITE)

    if draw_uniform:
        visualize_uniform_grid()
    if draw_adaptive:
        visualize_adaptive_grid()

    selected_grid = adaptive_grid if interpolate_with_adaptive else uniform_grid

    utils.draw_halo(screen, BLACKHOLE_CENTER, ENVIRONMENT_MAP_RADIUS, ENVIRONMENT_MAP_THICKNESS, ENVIRONMENT_MAP.lookup, resolution=ENVIRONMENT_MAP_RESOLUTION)
    pygame.gfxdraw.aacircle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)
    pygame.gfxdraw.filled_circle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)
    utils.draw_halo(screen, CAMERA_POSITION, CAMERA_RADIUS, CAMERA_THICKNESS, lambda phi: interpolate_hue(phi, selected_grid), resolution=CAMERA_RESOLUTION)

    manager.update(time_delta)
    manager.draw_ui(screen)
    pygame.display.flip()



