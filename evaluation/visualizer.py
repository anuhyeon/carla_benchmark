import numpy as np
import pygame


PANEL_WIDTH = 360


def setup(width, height, planner_width):
    pygame.init()
    screen = pygame.display.set_mode((width + planner_width + PANEL_WIDTH, height))
    pygame.display.set_caption("CARLA Policy Rollout")
    return screen, pygame.font.SysFont("monospace", 16, bold=True), planner_width


def draw_route(screen, font, rect, targets, target_index, position, path):
    x, y, width, height = rect
    route = np.asarray(targets)[:, :2]
    path = np.asarray(path)[:, :2]
    groups = [route, np.asarray(position)[None, :2]]
    if len(path):
        groups.append(path)
    points = np.concatenate(groups)
    low = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - low, 1.0)
    pad = 28
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])

    def project(point):
        px = x + pad + (point[0] - low[0]) * scale
        py = y + height - pad - (point[1] - low[1]) * scale
        return int(px), int(py)

    pygame.draw.rect(screen, (31, 35, 42), rect, border_radius=4)
    screen.blit(font.render("subgoal map", True, (115, 230, 185)), (x + 8, y + 6))

    route_points = [project(point) for point in route]
    if len(route_points) > 1:
        pygame.draw.lines(screen, (90, 110, 145), False, route_points, 2)
    for index, point in enumerate(route_points):
        color = (100, 108, 120) if index < target_index else (92, 170, 235)
        if index == target_index:
            color = (245, 205, 115)
        pygame.draw.circle(screen, color, point, 5)

    if len(path) > 1:
        pygame.draw.lines(
            screen, (245, 130, 90), False, [project(point) for point in path], 2
        )
    pygame.draw.circle(screen, (115, 230, 185), project(position), 7)


def draw(
    screen,
    font,
    planner_width,
    observation,
    targets,
    target_index,
    path,
    policy,
    prediction,
    elapsed,
    timeout,
    collisions,
):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
            return False

    rgb = observation["rgb"]
    screen.fill((18, 20, 24))
    screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
    screen.blit(font.render("Front RGB", True, (255, 255, 255)), (8, 8))

    if planner_width:
        planner = prediction["planner_visualization"]
        screen.blit(
            pygame.surfarray.make_surface(planner.swapaxes(0, 1)),
            (rgb.shape[1], 0),
        )
        screen.blit(
            font.render("GENIE Plan", True, (255, 255, 255)),
            (rgb.shape[1] + 8, 8),
        )

    position = observation["actor_position"]
    distance = np.linalg.norm(position[:2] - targets[target_index][:2])
    lines = [
        f"policy: {policy}",
        f"time: {elapsed:.1f}/{timeout:.1f}s",
        f"subgoal: {target_index + 1}/{len(targets)}",
        f"distance: {distance:.2f}m",
        f"speed: {observation['speed']:.2f}m/s",
        f"collisions: {collisions}",
        f"status: {prediction['status']}",
        "Q / ESC: quit",
    ]
    panel_x = rgb.shape[1] + planner_width + 16
    for index, text in enumerate(lines):
        screen.blit(font.render(text, True, (235, 238, 242)), (panel_x, 16 + index * 22))

    draw_route(
        screen,
        font,
        (panel_x, 210, PANEL_WIDTH - 32, rgb.shape[0] - 226),
        targets,
        target_index,
        position,
        path,
    )
    pygame.display.flip()
    return True


def close():
    pygame.quit()
