import multiprocessing
import multiprocessing.queues
import os
import random
import json
import sys
from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from bayes_opt import BayesianOptimization
from bayes_opt.event import Events
from bayes_opt.logger import JSONLogger
from bayes_opt.util import load_logs, UtilityFunction
from matplotlib.backends.backend_agg import FigureCanvasAgg

KAPPA = 2.576  # the default is 2.576, 10-30 is explorative, set to 0 for exploitation, rerun after changes
CIRCLES_PER_RUN = 20
LOADING = True


# TABLET_WIDTH = 224
# TABLET_HEIGHT = 148

def convert_to_tablet_coordinates(area_width, area_height, center_x, center_y, config):
    area_width = area_width * config['tablet_width'] / config['res_width']
    area_height = area_height * config['tablet_height'] / config['res_height']
    center_x = center_x * config['tablet_width'] / config['res_width']
    center_y = center_y * config['tablet_height'] / config['res_height']
    return area_width, area_height, center_x, center_y


def convert_to_pixel_coordinates(area_width, area_height, center_x, center_y, config):
    area_width = area_width * config['res_width'] / config['tablet_width']
    area_height = area_height * config['res_height'] / config['tablet_height']
    center_x = center_x * config['res_width'] / config['tablet_width']
    center_y = center_y * config['res_height'] / config['tablet_height']
    return area_width, area_height, center_x, center_y


def remap_cursor_position(x, y, area_width, area_height, center_x, center_y, res_width, res_height):
    mapped_x = (x - center_x) * res_width / area_width + res_width / 2
    mapped_y = (y - center_y) * res_height / area_height + res_height / 2
    return mapped_x, mapped_y


def plot_results(results, config):
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    fig.tight_layout(pad=3.0)
    ax = ax.flatten()
    area_widths, area_heights, center_xs, center_ys = convert_to_tablet_coordinates(np.array(results['area_width']),
                                                                                    np.array(results['area_height']),
                                                                                    np.array(results['center_x']),
                                                                                    np.array(results['center_y']),
                                                                                    config)
    y = np.array(results['target'])
    results = {'area_width': area_widths, 'area_height': area_heights, 'center_x': center_xs, 'center_y': center_ys}
    i = 0
    for param, x in results.items():
        ax[i].scatter(x, y, c=y, cmap='viridis_r')
        ax[i].set_title(f"Optimization results for {param}")
        ax[i].set_xlabel(param)
        ax[i].set_ylabel("Score")
        i += 1

    # 2D heatmap of the score as a function of the two parameters
    ax[4].set_title("Score heatmap")
    ax[4].set_xlabel("Area width")
    ax[4].set_ylabel("Area height")
    # blue to red cmap
    ax[4].scatter(results['area_width'], results['area_height'], c=y, cmap='viridis_r')

    # 2D heatmap of the score as a function of the two parameters
    ax[5].set_title("Score heatmap")
    ax[5].set_xlabel("Center X")
    ax[5].set_ylabel("Center Y")
    ax[5].scatter(results['center_x'], results['center_y'], c=y, cmap='viridis_r')

    # Convert the matplotlib plot into a raw RGB image data string
    canvas = FigureCanvasAgg(fig)
    canvas.figure.set_dpi(50)
    canvas.draw()
    width, height = canvas.get_width_height()
    raw_data = canvas.buffer_rgba()

    # Close the matplotlib figure to free up memory

    # Convert the raw RGB image data string into a Pygame surface
    plot_image = pygame.image.frombuffer(raw_data, (width, height), "RGBA")
    plt.close(fig)
    return plot_image


def run_game(params, results, optimal_params, screen, font, config, surface_plot):
    area_width = params['area_width']
    area_height = params['area_height']
    center_x = params['center_x']
    center_y = params['center_y']
    optimal_area_width = optimal_params['area_width']
    optimal_area_height = optimal_params['area_height']
    optimal_center_x = optimal_params['center_x']
    optimal_center_y = optimal_params['center_y']
    optimal_area_width, optimal_area_height, optimal_center_x, optimal_center_y = convert_to_tablet_coordinates(
        optimal_area_width, optimal_area_height, optimal_center_x, optimal_center_y, config)
    radius = 65
    cursor_radius = 20
    color = (0, 0, 255)  # blue
    next_color = (0, 255, 0)  # green
    colors = [color, next_color]
    if fail_bounds(params, config)[0]:
        return 0, True
    score = 0
    (tablet_area_width, tablet_area_height, tablet_center_x, tablet_center_y) = convert_to_tablet_coordinates(
        area_width,
        area_height,
        center_x,
        center_y, config)

    prev_x, prev_y = None, None
    results_plot_image = plot_results(results, config)
    if surface_plot is not None:
        surface_plot_image = plot_surfaces(surface_plot[0], surface_plot[1])
    xs, ys = [], []
    for i in range(CIRCLES_PER_RUN + 1):
        xs.append(random.randint(420 + radius, 2560 - 420 - radius))
        ys.append(random.randint(radius, 1440 - radius))
    for run in range(CIRCLES_PER_RUN + 1):
        x = xs[run]
        y = ys[run]

        running = True
        start_time = time()
        while running:
            mouse_pos = pygame.mouse.get_pos()
            cursor_pos = remap_cursor_position(mouse_pos[0], mouse_pos[1], area_width, area_height, center_x, center_y,
                                               config['res_width'], config['res_height'])

            screen.fill((0, 0, 0))
            if run < CIRCLES_PER_RUN:
                pygame.draw.circle(screen, colors[(run + 1) % len(colors)], (xs[run + 1], ys[run + 1]), radius)
            pygame.draw.circle(screen, colors[run % len(colors)], (x, y), radius)
            pygame.draw.circle(screen, (0, 0, 0), (x, y), radius // 3)
            pygame.draw.circle(screen, (255, 0, 0), cursor_pos, cursor_radius)

            screen.blit(results_plot_image, (0, config['res_height'] // 2 - results_plot_image.get_height() // 2))
            if surface_plot is not None:
                screen.blit(surface_plot_image, (config['res_width']-results_plot_image.get_width(), config['res_height'] // 2 - results_plot_image.get_height() // 2))
            text = font.render(
                f"Area: {tablet_area_width:.5f}mm x {tablet_area_height:.5f}mm, Center: {tablet_center_x:.5f}mm, {tablet_center_y:.5f}mm",
                True,
                (255, 255, 255))
            screen.blit(text, (10, 10))
            text1 = font.render(
                f"Optimal area: {optimal_area_width:.5f}mm x {optimal_area_height:.5f}mm, Center: {optimal_center_x:.5f}mm, {optimal_center_y:.5f}mm",
                True, (255, 255, 255))
            screen.blit(text1, (10, 30))
            text0 = font.render(f"Press Z or X on the circle with a black center as fast as you can (Q to quit)", True,
                                (255, 255, 255))
            screen.blit(text0, (10, 70))
            if run == 0:
                text5 = font.render(f"New run! (Hit the first circle to start or press q to quit)", True,
                                    (255, 255, 255))
                screen.blit(text5, (10, 90))

            text2 = font.render(f"Score: {score / run if run > 0 else 0}", True, (255, 255, 255))
            screen.blit(text2, (config['res_width'] - 200, 90))
            text3 = font.render(f"Mouse: {mouse_pos[0]}, {mouse_pos[1]}", True, (255, 255, 255))
            screen.blit(text3, (config['res_width'] - 200, 30))
            text4 = font.render(f"Cursor: {int(cursor_pos[0])}, {int(cursor_pos[1])}", True, (255, 255, 255))
            screen.blit(text4, (config['res_width'] - 200, 50))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None, False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return None, False
                    if event.key == pygame.K_z or event.key == pygame.K_x:
                        if run > 0 and (x - cursor_pos[0]) ** 2 + (y - cursor_pos[1]) ** 2 <= radius ** 2:
                            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                            score += distance / (time() - start_time)
                        running = False
                        prev_x, prev_y = cursor_pos
    return score / CIRCLES_PER_RUN, True


def run_configurator(screen, font, config):
    for direction in ['top', 'right', 'bottom', 'left']:

        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            screen.fill((0, 0, 0))
            text = font.render(f"Move your pen tip to the farthest {direction} position and press Z", True,
                               (255, 255, 255))
            screen.blit(text, (
                config['res_width'] // 2 - text.get_width() // 2, config['res_height'] // 2 - text.get_height() // 2))
            text1 = font.render(f"Position: {mouse_pos[0]}, {mouse_pos[1]}", True, (255, 255, 255))
            screen.blit(text1, (10, 10))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_z:
                        running = False
                        if direction == 'top':
                            config['top'] = mouse_pos[1]
                        elif direction == 'right':
                            config['right'] = mouse_pos[0]
                        elif direction == 'bottom':
                            config['bottom'] = mouse_pos[1]
                        elif direction == 'left':
                            config['left'] = mouse_pos[0]
    return config


def fail_bounds(params, config):
    area_width = params['area_width']
    area_height = params['area_height']
    center_x = params['center_x']
    center_y = params['center_y']
    if not area_width / 2 + center_x - area_width * 420 / 2560 < config['right']:
        return True, 'right'
    if not area_height / 2 + center_y < config['bottom']:
        return True, 'bottom'
    if not center_x - area_width / 2 + area_width * 420 / 2560 > config['left']:
        return True, 'left'
    if not center_y - area_height / 2 > config['top']:
        return True, 'top'
    return False, None


def surface_plot_data(optimizer, pbounds, suggestion):
    gp = optimizer._gp
    area_width_range = np.linspace(pbounds['area_width'][0], pbounds['area_width'][1], 100)
    area_height_range = np.linspace(pbounds['area_height'][0], pbounds['area_height'][1], 100)
    center_x_range = np.linspace(pbounds['center_x'][0], pbounds['center_x'][1], 100)
    center_y_range = np.linspace(pbounds['center_y'][0], pbounds['center_y'][1], 100)
    # create area width x area height surface and use suggestion as center x and center y
    area_width, area_height = np.meshgrid(area_width_range, area_height_range)
    center_x = np.ones_like(area_width) * suggestion['center_x']
    center_y = np.ones_like(area_width) * suggestion['center_y']
    X = np.dstack((area_width, area_height, center_x, center_y))
    X_2d = X.reshape(-1, 4)
    area_predictions, area_std = gp.predict(X_2d, return_std=True)
    area_predictions = area_predictions.reshape(area_width.shape)
    area_std = area_std.reshape(area_width.shape)
    area_data = (area_width, area_height, area_predictions, area_std)

    # create area width x center x surface and use suggestion as area height and center y
    center_x, center_y = np.meshgrid(center_x_range, center_y_range)
    area_width = np.ones_like(center_x) * suggestion['area_width']
    area_height = np.ones_like(center_x) * suggestion['area_height']
    X = np.dstack((area_width, area_height, center_x, center_y))
    X_2d = X.reshape(-1, 4)
    center_predictions, center_std = gp.predict(X_2d, return_std=True)
    center_predictions = center_predictions.reshape(center_x.shape)
    center_std = center_std.reshape(center_x.shape)
    center_data = (center_x, center_y, center_predictions, center_std)

    return area_data, center_data


def plot_surfaces(area_data, center_data):
    fig = plt.figure(figsize=(8, 12))
    fig.tight_layout(pad=0.0)
    area_width, area_height, area_predictions, area_std = area_data
    center_x, center_y, center_predictions, center_std = center_data
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.plot_surface(area_width, area_height, area_predictions, cmap='viridis', alpha=1)
    # ax1.plot_surface(area_width, area_height, area_predictions + area_std, cmap='viridis', alpha=0.2)
    # ax1.plot_surface(area_width, area_height, area_predictions - area_std, cmap='viridis', alpha=0.2)
    ax1.set_xlabel('Area width')
    ax1.set_ylabel('Area height')
    ax1.set_zlabel('Score')
    ax1.view_init(elev=60)
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(center_x, center_y, center_predictions, cmap='viridis', alpha=1)
    # ax2.plot_surface(center_x, center_y, center_predictions + center_std, cmap='viridis', alpha=0.2)
    # ax2.plot_surface(center_x, center_y, center_predictions - center_std, cmap='viridis', alpha=0.2)
    ax2.set_xlabel('Center x')
    ax2.set_ylabel('Center y')
    ax2.set_zlabel('Score')
    ax2.view_init(elev=60)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.figure.set_dpi(50)
    canvas.draw()
    width, height = canvas.get_width_height()
    raw_data = canvas.buffer_rgba()

    # Close the matplotlib figure to free up memory

    # Convert the raw RGB image data string into a Pygame surface
    plot_image = pygame.image.frombuffer(raw_data, (width, height), "RGBA")
    plt.close(fig)
    return plot_image


def OptimizerWorker(suggestion_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, config,
                    acquisition_function, probe=None, suggestor=False):
    pbounds = {
        'area_width' : (350, (config['right'] - config['left']) * (2560 + 2 * 420) / 2560),
        'area_height': (250, config['bottom'] - config['top']),
        'center_x'   : (config['left'], config['right']),
        'center_y'   : (config['top'], config['bottom']),
    }
    optimizer = BayesianOptimization(
        f=run_game,
        pbounds=pbounds,
        allow_duplicate_points=True,
    )

    if 'data.json' in os.listdir() and LOADING:
        print("Loading previous logs...")
        load_logs(optimizer, logs=["./data.json"])

    if suggestor:
        logger = JSONLogger(path="./data.json", reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    kernel = kernels.Matern() + kernels.WhiteKernel()
    optimizer.set_gp_params(n_restarts_optimizer=3, normalize_y=True, kernel=kernel, alpha=5e-1)
    # # acquisition_function = UtilityFunction(kind="ucb", kappa=0)
    # acquisition_function_optimal = UtilityFunction(kind="ucb", kappa=0)
    # # acquisition_function = UtilityFunction(kind='ucb')
    # acquisition_function = UtilityFunction(kind='ei', xi=1e-2)
    optimizer._prime_subscriptions()
    optimizer.dispatch(Events.OPTIMIZATION_START)
    results_queue_ = []
    init_points = True if probe is not None else False

    while True:
        if not suggestion_queue.full():
            if probe is None:
                acquisition_function.update_params()
                suggestion = optimizer.suggest(acquisition_function)
            else:
                suggestion = probe
                probe = None
            suggestion_as_array = optimizer._space._as_array(suggestion)
            results = {}
            for i, param in enumerate(optimizer.space.keys):
                x = [res['params'][param] for res in optimizer.res]
                y = [res['target'] for res in optimizer.res]
                results[param] = x
                results['target'] = y
            if suggestor and fail_bounds(suggestion, config)[0]:
                results_queue.put((suggestion_as_array, 0, results))
                continue
            else:
                plot_data = None
                if suggestor:
                    plot_data = surface_plot_data(optimizer, pbounds, suggestion)
                suggestion_queue.put((suggestion, suggestion_as_array, results, plot_data))
        if not results_queue.empty():
            params, score, running = results_queue.get()
            if not running:
                break
            if init_points and len(results_queue_) < 30:
                results_queue_.append((params, score))
                if len(results_queue_) == 30:
                    for params, score in results_queue_:
                        optimizer.register(params=params, target=score)
                    init_points = False
                    results_queue_ = []
            else:
                optimizer._space.register(params, score)
                optimizer.dispatch(Events.OPTIMIZATION_STEP)
        sleep(0.1)
    suggestion_queue.cancel_join_thread()
    results_queue.cancel_join_thread()


def main():
    config = None
    if 'config.json' in os.listdir():
        with open('config.json', 'r') as f:
            config = json.load(f)

    probe = None
    if config is None:
        print("\n\n\n\n\n\nOpen OpenTabletDriver")
        print("Collecting normal area for reference")
        width_probe = float(input("Enter the width of your current tablet area in mm: "))
        height_probe = float(input("Height: "))
        center_x_probe = float(input("X: "))
        center_y_probe = float(input("Y: "))
        print("Disable lock aspect ratio and resize the tablet area to full area in the right click menu of OTD.")
        tablet_width = float(input("Enter the width of the full area in mm: "))
        tablet_height = float(input("Height: "))
        config = {
            'tablet_width' : tablet_width,
            'tablet_height': tablet_height,
            'res_width'    : 2560,
            'res_height'   : 1440,
        }
        area_width, area_height, center_x, center_y = convert_to_pixel_coordinates(width_probe, height_probe,
                                                                                   center_x_probe, center_y_probe,
                                                                                   config)
        probe = {
            'area_width' : area_width,
            'area_height': area_height,
            'center_x'   : center_x,
            'center_y'   : center_y,
        }
    # if probe is None:
    #     area_width, area_height, center_x, center_y = convert_to_pixel_coordinates(78, 52,
    #                                                                                158, 81,
    #                                                                                config)
    #     probe = {
    #         'area_width' : area_width,
    #         'area_height': area_height,
    #         'center_x'   : center_x,
    #         'center_y'   : center_y,
    #     }

    input("Make sure your tablet area is set to the full area and press enter to start the game.")
    print(probe)

    pygame.init()
    window_size = (config['res_width'], config['res_height'])
    screen = pygame.display.set_mode(window_size)
    font = pygame.font.Font(None, 24)
    pygame.mouse.set_visible(False)

    if 'top' not in config.keys():
        config = run_configurator(screen, font, config)
        with open('config.json', 'w') as f:
            json.dump(config, f)

    suggestion_queue = multiprocessing.Queue(maxsize=1)
    optimal_suggestion_queue = multiprocessing.Queue(maxsize=1)
    results_queue = multiprocessing.Queue()
    results_queue_ = multiprocessing.Queue()

    acquisition_function_optimal = UtilityFunction(kind="ucb", kappa=0)
    acquisition_function = UtilityFunction(kind='ei', xi=1e-2)

    optimal_optimizer_process = multiprocessing.Process(
        target=OptimizerWorker,
        args=(optimal_suggestion_queue, results_queue_, config, acquisition_function_optimal, None, True),
    )
    optimal_optimizer_process.daemon = True
    optimal_optimizer_process.start()

    optimizer_process = multiprocessing.Process(
        target=OptimizerWorker,
        args=(suggestion_queue, results_queue, config, acquisition_function, probe, True),
    )
    optimizer_process.daemon = True
    optimizer_process.start()

    while True:
        x_probe, x_probe_as_array, results, surface_plot = suggestion_queue.get()
        optimal_x_probe, optimal_x_probe_as_array, _, _ = optimal_suggestion_queue.get()
        score, running = run_game(x_probe, results, optimal_x_probe, screen, font, config, surface_plot)
        results_queue.put((x_probe_as_array, score, running))
        results_queue_.put((x_probe_as_array, score, running))
        if not running:
            break

    optimizer_process.join()
    optimal_optimizer_process.join()
    pygame.quit()


if __name__ == '__main__':
    import pygame

    main()
