import json
import os
import pygame
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from functools import partial
from TabletAreaOpt.utils import remap_cursor_position, convert_to_pixel_coordinates
from scipy.optimize import minimize

CIRCLES_PER_RUN = 50
BUFFER_SIZE = 500
DISTANCE_FACTOR = 1


def run_game(config, plots, screen, font, data, prev_mouse_pos, params, total_circles):
    area_width = params[0]
    area_height = params[1]
    center_x = params[2]
    center_y = params[3]
    rotation = params[4]
    area_width_pixel, area_height_pixel, center_x_pixel, center_y_pixel = convert_to_pixel_coordinates(area_width,
                                                                                                       area_height,
                                                                                                       center_x,
                                                                                                       center_y, config)
    display_width, display_height = config['display']['res_width'], config['display']['res_height']

    radius = 80 * display_width / 2560
    cursor_radius = 20 * display_width / 2560
    color = (0, 0, 255)
    next_color = (0, 255, 0)
    colors = [color, next_color]

    prev_x, prev_y = None, None
    xs, ys = [], []
    corners = [(420 + radius, radius), (420 + radius, 1440 - radius), (2560 - 420 - radius, 1440 - radius),
               (2560 - 420 - radius, radius)]
    for i in range(total_circles):
        if i == 0:
            xs.append(display_width / 2)
            ys.append(display_height / 2)
        elif i < total_circles - 8:
            xs.append(np.random.randint(420 * display_width / 2560 + radius,
                                        2560 - 420 * display_width / 2560 - radius))
            ys.append(np.random.randint(radius, display_height - radius))
        else:
            xs.append(corners[(i - (total_circles - 8)) % 4][0])
            ys.append(corners[(i - (total_circles - 8)) % 4][1])

    mouse_pos_ar = []
    circle_pos_ar = []
    param_plot_image = None
    if len(plots.param_history) > 0:
        param_plot_image = plots.plot_params()
        error_plot_image = plots.plot_errors()

    for run in range(total_circles):
        x = xs[run]
        y = ys[run]

        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            cursor_pos = remap_cursor_position(mouse_pos[0], mouse_pos[1], area_width_pixel, area_height_pixel,
                                               center_x_pixel, center_y_pixel, display_width, display_height, rotation)

            screen.fill((0, 0, 0))
            if param_plot_image is not None:
                screen.blit(param_plot_image, (10, display_height // 2 - param_plot_image.get_height() // 2))
                screen.blit(error_plot_image, (display_width - error_plot_image.get_width(),
                                               display_height // 2 - error_plot_image.get_height() // 2))
            pygame.draw.line(screen, (255, 255, 255), (420, 0), (420, 1440), 1)
            pygame.draw.line(screen, (255, 255, 255), (display_width - 420, 0),
                             (display_width - 420, 1440), 1)

            if prev_x is not None:
                pygame.draw.line(screen, (40, 40, 40), (prev_x, prev_y), (x, y), 2)
            if run < total_circles - 1:
                pygame.draw.line(screen, (40, 40, 40), (x, y), (xs[run + 1], ys[run + 1]), 5)
                pygame.draw.circle(screen, colors[(run + 1) % len(colors)], (xs[run + 1], ys[run + 1]), radius)

            pygame.draw.circle(screen, colors[run % len(colors)], (x, y), radius)
            pygame.draw.circle(screen, (0, 0, 0), (x, y), radius // 3)
            pygame.draw.circle(screen, (255, 0, 0), cursor_pos, cursor_radius)

            text = font.render(
                f"Area: {area_width:.4f} x {area_height:.4f} Center: {center_x:.4f}, {center_y:.4f} Rotation: {rotation:.4f}°",
                True,
                (255, 255, 255))
            screen.blit(text, (display_width // 2 - text.get_width() // 2, 0))
            if run == 0:
                text0 = font.render(f"Press Z or X on the circle with a black center (R to restart run/Q to quit)",
                                    True,
                                    (255, 255, 255))
                screen.blit(text0, (display_width // 2 - text0.get_width() // 2, 40))
            text3 = font.render(f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", True, (255, 255, 255))
            screen.blit(text3, (display_width - 220, 30))
            text4 = font.render(f"Cursor: {int(cursor_pos[0])}, {int(cursor_pos[1])}", True, (255, 255, 255))
            screen.blit(text4, (display_width - 220, 50))
            text6 = font.render(
                f"Circle: {run + 1}/{total_circles}",
                True, (255, 255, 255))
            screen.blit(text6, (display_width - 220, 70))
            text7 = font.render(
                f"Step: {data['total_steps']}",
                True, (255, 255, 255))
            screen.blit(text7, (display_width - 220, 90))
            text = font.render(
                f"Buffer: {0 if prev_mouse_pos is None else len(prev_mouse_pos)} / {BUFFER_SIZE}",
                True, (255, 255, 255))
            screen.blit(text, (display_width - 220, 110))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_z or event.key == pygame.K_x:
                        mouse_pos_ar.append(mouse_pos)
                        circle_pos_ar.append((x, y))

                        running = False
                        prev_x, prev_y = cursor_pos
                    elif event.key == pygame.K_r:
                        return run_game(config, plots, screen, font, data, prev_mouse_pos, params, total_circles)
                    elif event.key == pygame.K_q:
                        return None

    return mouse_pos_ar, circle_pos_ar


class Plotting:
    def __init__(self):
        self.param_history = []
        self.error_history = []
        self.reg_error_history = []

    def add_history(self, params):
        self.param_history.append(params)

    def add_error(self, error):
        self.error_history.append(error)

    def add_reg_error(self, error):
        self.reg_error_history.append(error)

    def plot_params(self):
        fig, ax = plt.subplots(5, 1, figsize=(4, 12))
        fig.tight_layout(pad=1.5)
        fig.set_facecolor('black')
        ax = ax.flatten()
        for i, name in enumerate(['Width', 'Height', 'X', 'Y', 'Rotation']):
            ax[i].plot([p[i] for p in self.param_history], color='white')
            ax[i].set_title(name, color='white')
            ax[i].set_facecolor('black')
            ax[i].xaxis.label.set_color('white')
            ax[i].yaxis.label.set_color('white')
            ax[i].tick_params(axis='x', colors='white')
            ax[i].tick_params(axis='y', colors='white')
            for spine in ['top', 'right', 'bottom', 'left']:
                ax[i].spines[spine].set_color('white')
        return self.fig_to_plot_image(fig)

    def plot_errors(self):
        fig, ax = plt.subplots(2, 1, figsize=(4, 4))
        fig.tight_layout(pad=1.5)
        fig.set_facecolor('black')
        ax = ax.flatten()
        ax[0].plot(self.error_history, color='white')
        ax[0].set_title('Error (last run)', color='white')
        ax[0].set_facecolor('black')
        ax[0].xaxis.label.set_color('white')
        ax[0].yaxis.label.set_color('white')
        ax[0].tick_params(axis='x', colors='white')
        ax[0].tick_params(axis='y', colors='white')
        for spine in ['top', 'right', 'bottom', 'left']:
            ax[0].spines[spine].set_color('white')

        ax[1].plot(self.reg_error_history, color='white')
        ax[1].set_title('Regression Error (buffer)', color='white')
        ax[1].set_facecolor('black')
        ax[1].xaxis.label.set_color('white')
        ax[1].yaxis.label.set_color('white')
        ax[1].tick_params(axis='x', colors='white')
        ax[1].tick_params(axis='y', colors='white')
        for spine in ['top', 'right', 'bottom', 'left']:
            ax[1].spines[spine].set_color('white')

        return self.fig_to_plot_image(fig)

    @staticmethod
    def fig_to_plot_image(fig):
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        raw_data = canvas.buffer_rgba().tobytes()
        plt.close(fig)
        plot_image = pygame.image.frombuffer(raw_data, (width, height), 'RGBA')
        return plot_image


def run_configurator():
    config = {'display': {'res_width': 2560, 'res_height': 1440}}

    tablet = {}
    probe = {}
    if input('Would you like to enter a starting area? Otherwise will start with full area (y/n)') == 'y':
        probe['width'] = float(input('Enter current width: '))
        probe['height'] = float(input('Enter current height: '))
        probe['center_x'] = float(input('Enter current x: '))
        probe['center_y'] = float(input('Enter current y: '))
        probe['rotation'] = float(input('Enter current rotation: '))

    input('Resize your current area to full area and press enter...')
    tablet['width'] = float(input('Enter full area width: '))
    tablet['height'] = float(input('Enter full area  height: '))
    tablet['center_x'] = tablet['width'] / 2
    tablet['center_y'] = tablet['height'] / 2
    if 'width' not in probe:
        probe['width'] = tablet['width']
        probe['height'] = tablet['height']
        probe['center_x'] = tablet['width'] / 2
        probe['center_y'] = tablet['height'] / 2
        probe['rotation'] = 0
    print(
        f"Current area: {probe['width']} x {probe['height']} at {probe['center_x']}, {probe['center_y']} rotation {probe['rotation']}")
    print(f"Full area: {tablet['width']} x {tablet['height']}")
    if input("Is this correct (y/n)?") == 'y':
        config['tablet'] = tablet
        config['probe'] = probe
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        return config
    else:
        return run_configurator()


def objective_function(x, mouse_pos_ar, circle_pos_ar, config, radius=80, mean=True):
    if type(x) == list or type(x) == tuple:
        x = np.array(x)
    if x.ndim == 1:
        area_width, area_height, center_x, center_y, rotation = x[0, None, None], x[1, None, None], x[
            2, None, None], x[3, None, None], x[4, None, None]
    else:
        area_width, area_height, center_x, center_y, rotation = (
            x[:, 0, None], x[:, 1, None], x[:, 2, None], x[:, 3, None], x[:, 4, None])
    area_width, area_height, center_x, center_y = convert_to_pixel_coordinates(area_width, area_height, center_x,
                                                                               center_y, config)

    predicted_cursor_pos = remap_cursor_position(mouse_pos_ar[:, 0], mouse_pos_ar[:, 1], area_width, area_height,
                                                 center_x,
                                                 center_y, config['display']['res_width'],
                                                 config['display']['res_height'],
                                                 rotation)
    predicted_cursor_pos = np.stack((predicted_cursor_pos[0], predicted_cursor_pos[1]), axis=-1)
    if circle_pos_ar.ndim == 2:
        circle_pos_ar = circle_pos_ar[None, :, :]
    dist = np.linalg.norm(predicted_cursor_pos - circle_pos_ar, axis=2) / radius
    dist_from_center = np.linalg.norm(circle_pos_ar - np.array([config['display']['res_width'] / 2,
                                                                config['display']['res_height'] / 2])[None, None, :],
                                      axis=2) / radius
    weight = 1 + DISTANCE_FACTOR * dist_from_center
    if mean:
        return np.mean((dist ** 2) * weight, 1)
    else:
        return (dist ** 2) * weight


def main():
    if 'config.json' not in os.listdir():
        config = run_configurator()
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)

    if 'data.json' not in os.listdir():
        data = {'last_params': {}, 'total_steps': 0, 'mouse_pos': [], 'circle_pos': []}
        first_param = (config['probe']['width'], config['probe']['height'], config['probe']['center_x'],
                       config['probe']['center_y'], config['probe']['rotation'])
    else:
        with open('data.json', 'r') as f:
            data = json.load(f)
        first_param = (data['last_params']['width'], data['last_params']['height'], data['last_params']['center_x'],
                       data['last_params']['center_y'], data['last_params']['rotation'])

    bounds = np.array([(0, config['tablet']['width']), (0, config['tablet']['height']), (0, config['tablet']['width']),
                       (0, config['tablet']['height']), (-60, 60)])

    input("Make sure your tablet area is set to the full area and press enter to start the game.")

    pygame.init()
    window_size = (config['display']['res_width'], config['display']['res_height'])
    screen = pygame.display.set_mode(window_size)
    font = pygame.font.Font(None, 24)
    pygame.mouse.set_visible(False)
    plots = Plotting()

    run = 0
    mouse_pos, circle_pos = None, None
    prev_mouse_pos, prev_circle_pos = None, None
    if len(data['mouse_pos']) > 0:
        prev_mouse_pos = np.array(data['mouse_pos'])
        prev_circle_pos = np.array(data['circle_pos'])
    while True:
        if run == 0:
            params = first_param
        else:
            obj_function = partial(objective_function, mouse_pos_ar=mouse_pos, circle_pos_ar=circle_pos, config=config)
            params = None
            y_min = 1e6
            for _ in range(2):
                x_test = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(1000, 5))
                for i in range(100):
                    ys = obj_function(x_test)
                    x_test = x_test[ys.argsort()][:500]
                    res = minimize(obj_function, x_test[0], bounds=bounds, method='L-BFGS-B')
                    if res.fun < y_min:
                        y_min = res.fun
                        params = res.x
                        print(f"{i}: {y_min}, {params}")
                    x_test = np.random.normal(res.x, x_test.std(0), size=(1000, 5))
            plots.add_history(params)
            plots.add_reg_error(y_min)
            # params[0] *= 1.01
            # params[1] *= 1.01
            data['last_params'] = {'width'   : params[0],
                                   'height'  : params[1],
                                   'center_x': params[2],
                                   'center_y': params[3],
                                   'rotation': params[4]}
            with open('data.json', 'w') as f:
                json.dump(data, f)
        results = run_game(config, plots, screen, font, data, prev_mouse_pos, params, CIRCLES_PER_RUN)
        if results is None:
            break
        mouse_pos, circle_pos = results
        data['mouse_pos'].extend(mouse_pos)
        data['circle_pos'].extend(circle_pos)

        mouse_pos, circle_pos = np.array(mouse_pos), np.array(circle_pos)
        error = objective_function(params, mouse_pos, circle_pos, config=config, mean=True).item()
        plots.add_error(error)
        # mouse_pos, circle_pos = (mouse_pos[errors.argsort()][CIRCLES_PER_RUN // 2:],
        #                          circle_pos[errors.argsort()][CIRCLES_PER_RUN // 2:])
        run += 1
        data['total_steps'] += 1

        # if error <= 0.01 and prev_mouse_pos is not None:
        if prev_mouse_pos is not None:
            mouse_pos = np.concatenate((prev_mouse_pos, mouse_pos))[-BUFFER_SIZE:]
            circle_pos = np.concatenate((prev_circle_pos, circle_pos))[-BUFFER_SIZE:]
        prev_mouse_pos, prev_circle_pos = mouse_pos, circle_pos


if __name__ == '__main__':
    main()
