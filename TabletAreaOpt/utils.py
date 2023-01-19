import numpy as np


def convert_to_tablet_coordinates(area_width, area_height, center_x, center_y, config):
    area_width = area_width * config['tablet']['width'] / config['display']['res_width']
    area_height = area_height * config['tablet']['height'] / config['display']['res_height']
    center_x = center_x * config['tablet']['width'] / config['display']['res_width']
    center_y = center_y * config['tablet']['height'] / config['display']['res_height']
    return area_width, area_height, center_x, center_y


def convert_to_pixel_coordinates(area_width, area_height, center_x, center_y, config):
    area_width = area_width * config['display']['res_width'] / config['tablet']['width']
    area_height = area_height * config['display']['res_height'] / config['tablet']['height']
    center_x = center_x * config['display']['res_width'] / config['tablet']['width']
    center_y = center_y * config['display']['res_height'] / config['tablet']['height']
    return area_width, area_height, center_x, center_y


def rotate_point(x, y, x1, y1, angle):
    angle = -angle
    angle = np.deg2rad(angle)
    c, s = np.cos(angle), np.sin(angle)
    x = x - x1
    y = y - y1
    x_new = x * c - y * s + x1
    y_new = x * s + y * c + y1
    return x_new, y_new


def remap_cursor_position(x, y, area_width, area_height, center_x, center_y, res_width, res_height, rotation=0):
    x, y = rotate_point(x, y, center_x, center_y, rotation)
    mapped_x = (x - center_x) * res_width / area_width + res_width / 2
    mapped_y = (y - center_y) * res_height / area_height + res_height / 2
    return mapped_x, mapped_y


def reverse_cursor_position(x, y, area_width, area_height, center_x, center_y, res_width, res_height, rotation=0):
    x = (x - res_width / 2) * area_width / res_width + center_x
    y = (y - res_height / 2) * area_height / res_height + center_y
    x, y = rotate_point(x, y, center_x, center_y, -rotation)
    return x, y


def get_initial_config():
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
    config['probes'] = [probe]
    return config
