import pygame
import pygame.gfxdraw
import sys
import numpy as np
import utils
import schwarzchild2D
from environment_map import EnvironmentMapPolar


pygame.init()

WIDTH, HEIGHT = 1280, 720
DISTANCE_SCALE = 0.1
VELOCITY_SCALE = 1e-3
BLACKHOLE_CENTER = (WIDTH // 2, HEIGHT // 2)

START_TIME, END_TIME = 0, 1e10
ENVIRONMENT_MAP_THICKNESS = 10
ENVIRONMENT_MAP_RADIUS = min(WIDTH, HEIGHT)//2 - ENVIRONMENT_MAP_THICKNESS
ENVIRONMENT_MAP = EnvironmentMapPolar(np.linspace(0,1,num=10000, endpoint=False))

blakchole = schwarzchild2D.Schwarzchild2D(1)
blackhole_radius_pixels = int(blakchole.r_S / DISTANCE_SCALE)


screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Schwarzchild Explorer")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
screen.fill(WHITE)


drawing = False
start_pos = None
end_pos = None

trajectory_screen = None


clock = pygame.time.Clock()


while True:
    screen.fill(WHITE)
    utils.draw_environment_map(screen, BLACKHOLE_CENTER, ENVIRONMENT_MAP_RADIUS, ENVIRONMENT_MAP_THICKNESS, ENVIRONMENT_MAP.lookup, resolution=100)
    pygame.gfxdraw.aacircle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)
    pygame.gfxdraw.filled_circle(screen, BLACKHOLE_CENTER[0], BLACKHOLE_CENTER[1], blackhole_radius_pixels, BLACK)


    if trajectory_screen:
        pygame.draw.aalines(screen, BLACK, False, trajectory_screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            start_pos = event.pos
            end_pos = event.pos
            trajectory_screen = None  # clear previous trajectory


        elif event.type == pygame.MOUSEMOTION and drawing:
            end_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP and drawing:
            end_pos = event.pos
            drawing = False  # stop drawing arrow

            direction = np.array([end_pos[0] - start_pos[0], start_pos[1] - end_pos[1]])


            initial_cartesian_velocity = direction * VELOCITY_SCALE
            initial_cartesian_position = utils.screen_to_blackhole_coords(start_pos, WIDTH, HEIGHT, DISTANCE_SCALE)
            initial_polar_position = schwarzchild2D.cartesian_to_polar(initial_cartesian_position, batched=False)
            initial_polar_velocity = schwarzchild2D.polar_jacobian(initial_cartesian_position, batched=False) @ direction
            initial_conditions = np.array([
                initial_polar_position[0],  # r
                initial_polar_position[1],  # phi
                initial_polar_velocity[0],  # v_r
                initial_polar_velocity[1]   # v_phi
            ])
            t, trajectory_polar = blakchole.simulate(initial_conditions, start_time=START_TIME, end_time=END_TIME, batched=False)
            
            trajectory_cartesian = schwarzchild2D.polar_to_cartesian(trajectory_polar, batched=True)
            trajectory_screen = [utils.blackhole_coords_to_screen_coords(pos, WIDTH, HEIGHT, DISTANCE_SCALE) for pos in trajectory_cartesian]



    if drawing and start_pos and end_pos:
        utils.draw_arrow(screen, start_pos, end_pos)

    pygame.display.flip()
    clock.tick(60)


