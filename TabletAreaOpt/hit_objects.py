import os
import numpy as np


def convert_to_current_coordinates_(hit_objects, config):
    hit_objects[:, 0] = hit_objects[:, 0] / 640 * config['playfield']['width'] + config['playfield']['left']
    hit_objects[:, 1] = hit_objects[:, 1] / 480 * config['playfield']['height'] + config['playfield']['top']
    return hit_objects


def parse_hit_objects(osu_file, config):
    hit_objects_tag = False
    hit_objects = []
    with open(osu_file, "r", encoding="utf-8") as f:
        for line in f:
            if '[HitObjects]' in line:
                hit_objects_tag = True
                continue
            if hit_objects_tag:
                hit_objects.append(line.split(",")[:2])
                if len(line.strip()) == 0:
                    break
    return convert_to_current_coordinates_(np.array(hit_objects, dtype=np.float32), config)


def get_osu_filepaths(osu_song_dir):
    osu_filepaths = []
    for folder in os.listdir(osu_song_dir):
        folder_path = os.path.join(osu_song_dir, folder)
        if os.path.isdir(folder_path):
            for osu_file in os.listdir(folder_path):
                if osu_file.endswith(".osu"):
                    osu_filepaths.append(os.path.join(folder_path, osu_file))
    return osu_filepaths
