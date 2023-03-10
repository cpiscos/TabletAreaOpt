import json
import os
import time

import pygame
import pygame.gfxdraw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from functools import partial
from TabletAreaOpt.utils import remap_cursor_position, convert_to_pixel_coordinates
from TabletAreaOpt.hit_objects import get_osu_filepaths, parse_hit_objects
from scipy.optimize import minimize

DEBUG = False
CIRCLES_PER_RUN = 30 if not DEBUG else 10
BUFFER_SIZE = 600
DISTANCE_FACTOR = 1
OPTIMIZATION_ITERATIONS = 1
NUM_RESTARTS = 3
OSU_SONG_DIR = fr'C:\Users\{os.getlogin()}\AppData\Local\osu!\Songs'
OTD_PRESET_PATH = fr'C:\Users\{os.getlogin()}\AppData\Local\OpenTabletDriver\Presets\opt.json'


class Game:
    def __init__(self, config, CS=4):
        pygame.init()
        # window_size = (config['display']['res_width'], config['display']['res_height'])
        # self.screen = pygame.display.set_mode(window_size)
        # fullscreen
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        assert self.screen.get_size() == (config['display']['res_width'], config['display']['res_height'])
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 100)
        pygame.mouse.set_visible(False)

        self.config = config

        self.prev_x = None
        self.prev_y = None

        self.plots = Plotting()
        self.display_width, self.display_height = config['display']['res_width'], config['display']['res_height']
        self.radius = (54.4 - 4.48 * CS) * self.config['playfield']['width'] / 640
        self.cursor_radius = 20 * self.display_width / 2560
        circle_color = (0, 0, 255)
        next_circle_color = (0, 255, 0)
        self.colors = [circle_color, next_circle_color]
        self.init_text = self.font.render(
            f"Press Z or X on the circle with a black center. Aim for the center. (R to restart run/Q to quit/P to export)",
            True,
            (255, 255, 255))
        self.clock = pygame.time.Clock()
        self.osu_filepaths = get_osu_filepaths(OSU_SONG_DIR)

    def run_game(self, data, prev_mouse_pos, params, total_circles, random_gen=False):
        start_time = time.time()
        area_width = params[0]
        area_height = params[1]
        center_x = params[2]
        center_y = params[3]
        rotation = params[4]
        area_width_pixel, area_height_pixel, center_x_pixel, center_y_pixel = convert_to_pixel_coordinates(area_width,
                                                                                                           area_height,
                                                                                                           center_x,
                                                                                                           center_y,
                                                                                                           self.config)
        area_text = self.font.render(
            f"Area: {area_width:.4f} x {area_height:.4f} Center: {center_x:.4f}, {center_y:.4f} Rotation: {rotation:.4f}??",
            True,
            (255, 255, 255))
        total_steps_text = self.font.render(
            f"Step: {data['total_steps']}",
            True, (255, 255, 255))
        buffer_size_text = self.font.render(
            f"Buffer: {0 if prev_mouse_pos is None else len(prev_mouse_pos)} / {BUFFER_SIZE}",
            True, (255, 255, 255))
        circles_since_start_text = self.font.render(
            f"Total circles: {len(data['mouse_pos'])}",
            True, (255, 255, 255))

        xs, ys = [], []
        # counterclockwise rotation from top left
        corners = [(self.config['playfield']['left'], self.config['playfield']['top']),
                   (self.config['playfield']['left'], self.config['playfield']['bottom']),
                   (self.config['playfield']['right'], self.config['playfield']['bottom']),
                   (self.config['playfield']['right'], self.config['playfield']['top'])]

        hit_objects = None
        while hit_objects is None:
            try:
                idx = np.random.randint(len(self.osu_filepaths))
                hit_objects_ = parse_hit_objects(self.osu_filepaths[idx], self.config)
                # ensure angle between consecutive hit objects are all < 90 degrees

                if len(hit_objects_) < CIRCLES_PER_RUN:
                    self.osu_filepaths.pop(idx)
                    print('removed, now:', len(self.osu_filepaths))
                    continue
                idx = np.random.randint(len(hit_objects_) - CIRCLES_PER_RUN)
                hit_objects_ = hit_objects_[idx:idx + CIRCLES_PER_RUN]
                if len(np.unique(hit_objects_, axis=0)) != CIRCLES_PER_RUN:
                    continue
                hit_objects = hit_objects_.tolist()
            except Exception as e:
                print(e)
                continue

        for i in range(total_circles):
            if i == 0:
                xs.append(self.display_width / 2)
                ys.append(self.display_height - self.config['playfield']['height'] / 2)
            elif i < total_circles - 4:
                if random_gen:
                    xs.append(np.random.randint(self.config['playfield']['left'],
                                                self.config['playfield']['right']))
                    ys.append(np.random.randint(self.config['playfield']['top'],
                                                self.config['playfield']['bottom']))
                else:
                    xs.append(hit_objects[i - 1][0])
                    ys.append(hit_objects[i - 1][1])
            else:
                xs.append(corners[(i - (total_circles - 8)) % 4][0])
                ys.append(corners[(i - (total_circles - 8)) % 4][1])

        mouse_pos_ar = []
        circle_pos_ar = []
        param_plot_image = None
        error_plot_image = None
        if len(self.plots.param_history) > 0:
            param_plot_image = self.plots.plot_params()
            error_plot_image = self.plots.plot_errors()
        print("Setup time: ", time.time() - start_time)
        for run in range(total_circles):
            x = xs[run]
            y = ys[run]

            running = True
            while running:
                mouse_pos = pygame.mouse.get_pos()
                cursor_pos = remap_cursor_position(mouse_pos[0], mouse_pos[1], area_width_pixel, area_height_pixel,
                                                   center_x_pixel, center_y_pixel, self.display_width,
                                                   self.display_height,
                                                   rotation)
                mouse_pos_text = self.font.render(f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", True, (255, 255, 255))
                cursor_text = self.font.render(f"Cursor: {int(cursor_pos[0])}, {int(cursor_pos[1])}", True,
                                               (255, 255, 255))
                total_circles_text = self.font.render(
                    f"Circle: {run + 1}/{total_circles}",
                    True, (255, 255, 255))
                self.clock.tick()
                fps_text = self.font.render(
                    f"{self.clock.get_fps():.2f} FPS",
                    True, (255, 255, 255))

                self.screen.fill((0, 0, 0))
                if param_plot_image is not None:
                    self.screen.blit(param_plot_image,
                                     (self.config['playfield']['left'] - param_plot_image.get_width() - self.radius,
                                      self.display_height // 2 - param_plot_image.get_height() // 2))
                    self.screen.blit(error_plot_image, (self.config['playfield']['right'] + self.radius,
                                                        self.display_height // 2 - error_plot_image.get_height() // 2))
                # pygame.draw.line(self.screen, (255, 255, 255),
                #                  (self.config['playfield']['left'], self.config['playfield']['top']),
                #                  (self.config['playfield']['right'], self.config['playfield']['top']))
                # pygame.draw.line(self.screen, (255, 255, 255),
                #                  (self.config['playfield']['left'], self.config['playfield']['top']),
                #                  (self.config['playfield']['left'], self.config['playfield']['bottom']))
                # pygame.draw.line(self.screen, (255, 255, 255),
                #                  (self.config['playfield']['right'], self.config['playfield']['top']),
                #                  (self.config['playfield']['right'], self.config['playfield']['bottom']))
                #
                # if run < total_circles - 1:
                #     pygame.draw.line(self.screen, (40, 40, 40), (x, y), (xs[run + 1], ys[run + 1]), 5)
                #     pygame.draw.circle(self.screen, self.colors[(run + 1) % len(self.colors)],
                #                        (xs[run + 1], ys[run + 1]),
                #                        self.radius)
                #
                # pygame.draw.circle(self.screen, self.colors[run % len(self.colors)], (x, y), self.radius)
                # pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.radius // 3)
                # pygame.draw.circle(self.screen, (255, 0, 0), cursor_pos, self.cursor_radius)
                # pygame.draw.line(self.screen, (100, 0, 0), (cursor_pos[0], cursor_pos[1]), (x, y), 2)

                # with pygame.gfxdraw
                pygame.gfxdraw.line(self.screen, int(self.config['playfield']['left'] - self.radius),
                                    int(self.config['playfield']['top'] - self.radius),
                                    int(self.config['playfield']['right'] + self.radius),
                                    int(self.config['playfield']['top'] - self.radius),
                                    (255, 255, 255))
                pygame.gfxdraw.line(self.screen, int(self.config['playfield']['left'] - self.radius),
                                    int(self.config['playfield']['top'] - self.radius),
                                    int(self.config['playfield']['left'] - self.radius),
                                    int(self.config['playfield']['bottom'] + self.radius),
                                    (255, 255, 255))
                pygame.gfxdraw.line(self.screen, int(self.config['playfield']['right'] + self.radius),
                                    int(self.config['playfield']['top'] - self.radius),
                                    int(self.config['playfield']['right'] + self.radius),
                                    int(self.config['playfield']['bottom'] + self.radius),
                                    (255, 255, 255))
                pygame.gfxdraw.line(self.screen, int(self.config['playfield']['left'] - self.radius),
                                    int(self.config['playfield']['bottom'] + self.radius),
                                    int(self.config['playfield']['right'] + self.radius),
                                    int(self.config['playfield']['bottom'] + self.radius),
                                    (255, 255, 255))

                # if run < total_circles - 3:
                #     pygame.gfxdraw.line(self.screen, int(xs[run + 2]), int(ys[run + 2]), int(xs[run + 3]),
                #                         int(ys[run + 3]),
                #                         (40, 40, 40))
                # if run < total_circles - 3:
                #     if not (run % 3) == 0:
                #         pygame.gfxdraw.line(self.screen, int(xs[run + 2]), int(ys[run + 2]), int(xs[run + 3]),
                #                             int(ys[run + 3]),
                #                             (80, 80, 80))
                if run < total_circles - 2:
                    pygame.gfxdraw.line(self.screen, int(xs[run + 1]), int(ys[run + 1]), int(xs[run + 2]),
                                        int(ys[run + 2]),
                                        (160, 160, 160))
                if run < total_circles - 1:
                    pygame.gfxdraw.line(self.screen, int(x), int(y), int(xs[run + 1]), int(ys[run + 1]),
                                        (255, 255, 255))
                # if run < total_circles - 3:
                #     pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 3]), int(ys[run + 3]), int(self.radius),
                #                                  self.colors[((run + 3) // 2) % len(self.colors)])
                # if run < total_circles - 3:
                #     pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 3]), int(ys[run + 3]), int(self.radius),
                #                                  self.colors[((run + 3) // 3) % len(self.colors)])
                #     pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 3]), int(ys[run + 3]),
                #                                  int(self.radius // 1.5),
                #                                  (0, 0, 0))
                #     hit_3_text = self.large_font.render(str((run + 3) % 3 + 1), True, (255, 255, 255))
                #     hit_3_text_rect = hit_3_text.get_rect()
                #     hit_3_text_rect.center = (int(xs[run + 3]), int(ys[run + 3]))
                #     self.screen.blit(hit_3_text, hit_3_text_rect)
                #     pygame.gfxdraw.aacircle(self.screen, int(xs[run + 3]), int(ys[run + 3]), int(self.radius),
                #                             self.colors[((run + 3) // 3) % len(self.colors)])
                #     pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 3]), int(ys[run + 3]),
                #                                  int(self.radius * 1.1),
                #                                  (0, 0, 0, 210))
                if run < total_circles - 2:
                    pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 2]), int(ys[run + 2]), int(self.radius),
                                                 self.colors[((run + 2) // 3) % len(self.colors)])
                    pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 2]), int(ys[run + 2]),
                                                 int(self.radius // 1.5),
                                                 (0, 0, 0))
                    hit_2_text = self.large_font.render(str((run + 2) % 3 + 1), True, (255, 255, 255))
                    hit_2_text_rect = hit_2_text.get_rect()
                    hit_2_text_rect.center = (int(xs[run + 2]), int(ys[run + 2]))
                    self.screen.blit(hit_2_text, hit_2_text_rect)
                    pygame.gfxdraw.aacircle(self.screen, int(xs[run + 2]), int(ys[run + 2]), int(self.radius * 1.5),
                                            self.colors[((run + 2) // 3) % len(self.colors)])
                if run < total_circles - 1:
                    pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 1]), int(ys[run + 1]), int(self.radius),
                                                 self.colors[((run + 1) // 3) % len(self.colors)])
                    pygame.gfxdraw.filled_circle(self.screen, int(xs[run + 1]), int(ys[run + 1]),
                                                 int(self.radius // 1.5),
                                                 (0, 0, 0))
                    hit_1_text = self.large_font.render(str((run + 1) % 3 + 1), True, (255, 255, 255))
                    hit_1_text_rect = hit_1_text.get_rect()
                    hit_1_text_rect.center = (int(xs[run + 1]), int(ys[run + 1]))
                    self.screen.blit(hit_1_text, hit_1_text_rect)
                    pygame.gfxdraw.aacircle(self.screen, int(xs[run + 1]), int(ys[run + 1]), int(self.radius * 1.3),
                                            self.colors[((run + 1) // 3) % len(self.colors)])

                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(self.radius),
                                             self.colors[(run // 3) % len(self.colors)])
                # ring around the current circle
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(self.radius * 1.1),
                                        self.colors[(run // 3) % len(self.colors)])

                pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), int(self.radius // 1.5),
                                             (0, 0, 0))

                hit_text = self.large_font.render(str(run % 3 + 1), True, (255, 255, 255))
                hit_text_rect = hit_text.get_rect()
                hit_text_rect.center = (int(x), int(y))
                self.screen.blit(hit_text, hit_text_rect)
                # circle border
                pygame.gfxdraw.aacircle(self.screen, int(x), int(y), int(self.radius),
                                        self.colors[(run // 3) % len(self.colors)])
                pygame.gfxdraw.filled_circle(self.screen, int(cursor_pos[0]), int(cursor_pos[1]),
                                             int(self.cursor_radius), (255, 0, 0))
                # pygame.gfxdraw.line(self.screen, int(cursor_pos[0]), int(cursor_pos[1]), int(x), int(y), (255, 0, 0))

                self.screen.blit(area_text, (self.display_width // 2 - area_text.get_width() // 2, 0))
                self.screen.blit(self.init_text, (self.display_width // 2 - self.init_text.get_width() // 2, 40))
                self.screen.blit(mouse_pos_text, (self.display_width - 220, 30))
                self.screen.blit(cursor_text, (self.display_width - 220, 50))
                self.screen.blit(total_circles_text, (self.display_width - 220, 70))
                self.screen.blit(total_steps_text, (self.display_width - 220, 90))
                self.screen.blit(buffer_size_text, (self.display_width - 220, 110))
                self.screen.blit(circles_since_start_text, (self.display_width - 220, 130))
                self.screen.blit(fps_text, (self.display_width - 220, self.display_height - 30))
                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_z or event.key == pygame.K_x:
                            mouse_pos_ar.append(mouse_pos)
                            circle_pos_ar.append((x, y))

                            running = False
                            self.prev_x, self.prev_y = cursor_pos
                        elif event.key == pygame.K_r:
                            return self.run_game(data, prev_mouse_pos, params, total_circles)
                        elif event.key == pygame.K_q:
                            return None
                        elif event.key == pygame.K_p:
                            save_params_to_preset(config=self.config, params=params)
        return mouse_pos_ar, circle_pos_ar


class Plotting:
    def __init__(self):
        self.param_history = []
        self.error_history = []
        self.reg_error_history = []

    def add_param(self, params):
        self.param_history.append(params)

    def add_error(self, error):
        self.error_history.append(error)

    def add_reg_error(self, error):
        self.reg_error_history.append(error)

    def plot_params(self):
        fig, ax = plt.subplots(5, 1, figsize=(3.5, 12))
        fig.tight_layout(pad=2)
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
            ax[i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            for spine in ['top', 'right', 'bottom', 'left']:
                ax[i].spines[spine].set_color('white')
        return self.fig_to_plot_image(fig)

    def plot_errors(self):
        fig, ax = plt.subplots(2, 1, figsize=(3.5, 4))
        fig.tight_layout(pad=2)
        fig.set_facecolor('black')
        ax = ax.flatten()
        ax[0].plot(self.error_history, color='white')
        ax[0].set_title('Error (Last Run)', color='white')
        ax[0].set_facecolor('black')
        ax[0].xaxis.label.set_color('white')
        ax[0].yaxis.label.set_color('white')
        ax[0].tick_params(axis='x', colors='white')
        ax[0].tick_params(axis='y', colors='white')
        for spine in ['top', 'right', 'bottom', 'left']:
            ax[0].spines[spine].set_color('white')

        ax[1].plot(self.reg_error_history, color='white')
        ax[1].set_title('Regression Error (Buffer)', color='white')
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
        # noinspection PyTypedDict
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
            json.dump(config, f, indent=2)
        return config
    else:
        return run_configurator()


def objective_function(x, mouse_pos_ar, circle_pos_ar, config, radius=90, mean=True):
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
    dist = np.clip(dist - 1, 0, None)
    dist_from_center = np.linalg.norm(circle_pos_ar - np.array([config['display']['res_width'] / 2,
                                                                config['display']['res_height'] / 2])[None, None, :],
                                      axis=2) / radius
    weight = 1 + DISTANCE_FACTOR * dist_from_center
    if mean:
        return np.mean((dist ** 2) * weight, 1)
    else:
        return (dist ** 2) * weight


def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    if 'config.json' not in os.listdir():
        config = run_configurator()
    else:
        with open('config.json', 'r') as f:
            config = json.load(f)
        playfield = {}
        playfield['height'] = 0.80 * config['display']['res_height']
        playfield['width'] = 4 / 3 * playfield['height']
        playfield['top'] = (config['display']['res_height'] - playfield['height']) * 4 / 7
        playfield['bottom'] = playfield['top'] + playfield['height']

        playfield['left'] = (config['display']['res_width'] - playfield['width']) / 2
        playfield['right'] = playfield['left'] + playfield['width']

        config['playfield'] = playfield

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

    input("Make sure your tablet area is set to the full area (0 rotation) and press enter to start.")

    game = Game(config)

    mouse_pos, circle_pos = None, None
    prev_mouse_pos, prev_circle_pos = None, None
    start_time = time.time()
    if len(data['mouse_pos']) > 0:
        prev_mouse_pos = np.array(data['mouse_pos'])
        prev_circle_pos = np.array(data['circle_pos'])
        if len(prev_mouse_pos) > BUFFER_SIZE:
            p = (1 - CIRCLES_PER_RUN / BUFFER_SIZE) ** (np.arange(len(data['mouse_pos']))[::-1] / CIRCLES_PER_RUN)
            indices = np.random.choice(len(data['mouse_pos']), size=BUFFER_SIZE, replace=False,
                                       p=p / np.sum(p))
            prev_mouse_pos = prev_mouse_pos[indices]
            prev_circle_pos = prev_circle_pos[indices]

    while True:
        if mouse_pos is None:
            params = first_param
        else:
            obj_function = partial(objective_function, mouse_pos_ar=mouse_pos, circle_pos_ar=circle_pos, config=config,
                                   radius=game.radius)
            params = None
            y_min = 1e6
            for _ in range(NUM_RESTARTS):
                x_test = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(2000, 5))
                print("time 2: ", time.time() - start_time)
                for i in range(OPTIMIZATION_ITERATIONS):
                    ys = obj_function(x_test)
                    x_test = x_test[ys.argsort()]
                    res = minimize(obj_function, x_test[0], bounds=bounds, method='L-BFGS-B')
                    if res.fun < y_min:
                        y_min = res.fun
                        params = res.x
                        print(f"{i}: {y_min}, {params}")
                    if i < OPTIMIZATION_ITERATIONS - 1:
                        x_test = x_test[:500]
                        x_test = np.random.normal(res.x, x_test.std(0), size=(2000, 5))
            game.plots.add_param(params)
            game.plots.add_reg_error(y_min)
            data['last_params'] = {'width'   : params[0],
                                   'height'  : params[1],
                                   'center_x': params[2],
                                   'center_y': params[3],
                                   'rotation': params[4]}
            if not DEBUG:
                with open('data.json', 'w') as f:
                    json.dump(data, f)

        print("total time between runs: ", time.time() - start_time)
        results = game.run_game(data, prev_mouse_pos, params, CIRCLES_PER_RUN)
        start_time = time.time()
        if results is None:
            break
        mouse_pos, circle_pos = results
        data['mouse_pos'].extend(mouse_pos.copy())
        data['circle_pos'].extend(circle_pos.copy())

        mouse_pos, circle_pos = np.array(mouse_pos), np.array(circle_pos)
        error = objective_function(params, mouse_pos, circle_pos, config=config, mean=True).item()
        game.plots.add_error(error)
        data['total_steps'] += 1

        if prev_mouse_pos is not None:
            indices = np.random.choice(len(prev_mouse_pos), min(len(prev_mouse_pos), BUFFER_SIZE - CIRCLES_PER_RUN),
                                       replace=False)
            prev_mouse_pos = prev_mouse_pos[indices]
            prev_circle_pos = prev_circle_pos[indices]
            mouse_pos = np.concatenate((prev_mouse_pos, mouse_pos))
            circle_pos = np.concatenate((prev_circle_pos, circle_pos))
        prev_mouse_pos, prev_circle_pos = mouse_pos, circle_pos
        print("time 1: ", time.time() - start_time)


def save_params_to_preset(config, params):
    with open(OTD_PRESET_PATH, 'r') as f:
        otd_preset = json.load(f)
        profile = None
        for p in otd_preset['Profiles']:
            if p['Tablet'] == config['tablet']['name'] or len(otd_preset['Profiles']) == 1:
                profile = p
                break
    tablet_settings = profile['AbsoluteModeSettings']['Tablet']
    tablet_settings['Width'] = float(params[0])
    tablet_settings['Height'] = float(params[1])
    tablet_settings['X'] = float(params[2])
    tablet_settings['Y'] = float(params[3])
    tablet_settings['Rotation'] = float(params[4])
    with open(OTD_PRESET_PATH, 'w') as f:
        json.dump(otd_preset, f, indent=2)


if __name__ == '__main__':
    main()
