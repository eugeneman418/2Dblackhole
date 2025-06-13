import pygame
import pygame.gfxdraw
import sys
import math
import numpy as np
import utils
import schwarzchild2D

blakchole = schwarzchild2D.Schwarzchild2D(1)

pygame.init()

WIDTH, HEIGHT = 800, 600
distance_scale = 0.1
velocity_scale = 1e-3
blackhole_center = (WIDTH // 2, HEIGHT // 2)
radius_pixels = int(blakchole.r_S / distance_scale)
start_time, end_time = 0, 1e10

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


while True:
    screen.fill(WHITE)
    pygame.gfxdraw.aacircle(screen, blackhole_center[0], blackhole_center[1], radius_pixels, BLACK)
    pygame.gfxdraw.filled_circle(screen, blackhole_center[0], blackhole_center[1], radius_pixels, BLACK)


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


            initial_cartesian_velocity = direction * velocity_scale
            initial_cartesian_position = utils.screen_to_blackhole_coords(start_pos, WIDTH, HEIGHT, distance_scale)
            initial_polar_position = schwarzchild2D.cartesian_to_polar(initial_cartesian_position, batched=False)
            initial_polar_velocity = schwarzchild2D.polar_jacobian(initial_cartesian_position, batched=False) @ direction
            initial_conditions = np.array([
                initial_polar_position[0],  # r
                initial_polar_position[1],  # phi
                initial_polar_velocity[0],  # v_r
                initial_polar_velocity[1]   # v_phi
            ])
            t, trajectory_polar = blakchole.simulate(initial_conditions, start_time=0, end_time=100000, batched=False)
            print("number of steps:", len(t))
            trajectory_cartesian = schwarzchild2D.polar_to_cartesian(trajectory_polar, batched=True)
            trajectory_screen = [utils.blackhole_coords_to_screen_coords(pos, WIDTH, HEIGHT, distance_scale) for pos in trajectory_cartesian]



    if drawing and start_pos and end_pos:
        draw_arrow(screen, start_pos, end_pos)

    pygame.display.flip()
    clock.tick(60)


