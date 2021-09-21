import os
import shutil
import json
import numpy as np
import pandas as pd
from cv2 import cv2
import xml.etree.ElementTree as ET

def load_image(path):
    return cv2.imread(path)

def load_object_points(filepath):
    object_points_dict = {}
    with open(filepath) as f:
        res = json.load(f)

    for key, value in res.items():
        object_points_dict[key] = np.array(value, dtype=np.float32)
    return object_points_dict

def load_image_points(filedir):
    image_points_dict = []
    for filename in sorted(os.listdir(filedir)):
        temp_image_points_dict = {}
        with open(os.path.join(filedir, filename)) as f:
            res = json.load(f)

        for each in res["shapes"]:
            temp_image_points_dict[int(each["label"])] = np.array(each["points"][0], dtype=np.float32)
        image_points_dict.append(temp_image_points_dict)

    return image_points_dict

def load_polygon_points(path):
    """
    load the key points that margined the road area.
    """
    with open(path) as f:
        res = json.load(f)

    for each in res["shapes"]:
        if each["label"] == "road_area":
            return np.array(each["points"]).astype(np.int32)


def load_lane_points(path):
    """
    load the seperate line in the road area
    :param path:
    :return:
    """
    retval = []
    with open(path) as f:
        res = json.load(f)

    for each in res["shapes"]:
        if "lane" in each["label"]:
            retval.append(np.array(each["points"]).astype(np.int32))
    return retval

def load_length_mark_points(path, choice="real"):
    """
    load the points that stand for the length of the road.
    :param path:
    :return:
    """
    retval = []
    with open(path) as f:
        res = json.load(f)

    for each in res["shapes"]:
        if choice in each["label"]:
            retval.append(np.array(each["points"]).astype(np.int32))
    return retval






def load_certain_params_with_cartype_and_index(path, cartype):
    data = np.load(path)
    cartypes = data["cartype"]
    rvecs = data["best_rvec"]
    tvecs = data["best_tvec"]
    f = data["best_f"]

    for i in range(len(cartypes)):
        if cartype == cartypes[i]:
            return rvecs[i], tvecs[i], f[i]


def camera_param_iterator(rootpath):
    for each in os.listdir(rootpath):
        if "npz" not in each:
            continue
        
        filename = os.path.join(rootpath, each)
        print(filename)
        data = np.load(filename)
        rvecs = data["best_rvec"]
        tvecs = data["best_tvec"]
        fs = data["best_f"]
        losses = data["loss"]
        print(np.shape(rvecs))
        for item in zip(rvecs, tvecs, fs, losses):
            yield item

def load_bounding_box(path):
    """
    loading the box of vehicles segmented by YOLO_v5
    """
    text = pd.read_csv(path, delimiter=" ", names=["label", "x0", "y0", "x1", "y1", "junk"])
    return (text.dropna(axis=1).values)

def parse_config_file(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    for each in root:
        name = each[0].text
        object_points_path = each[1].text
        image_points_index_path = each[2].text
        max_label_nums = each[3].text
        montage_width = each[4].text
        montage_height = each[5].text
        yield name, object_points_path, image_points_index_path, max_label_nums, montage_width, montage_height

if __name__ == '__main__':
    filename = "camera_config.xml"
    for each in parse_config_file(filename):
        print(each)