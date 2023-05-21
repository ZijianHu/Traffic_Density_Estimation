import numpy as np
# from cv2 import cv2
import cv2


def generate_polygon_mask(points, fill=True):
    """
    generate a mask of the main road.
    """
    canvas = np.zeros((240, 320, 3), dtype="uint8")
    if fill:
        cv2.fillPoly(canvas, pts=[points], color=(255, 255, 255), lineType=cv2.LINE_AA)
    else:
        cv2.polylines(canvas, pts=[points], isClosed=True, color=(0, 0, 255), thickness=3)
    return canvas


def softmax(x, t=1):
    """
    softmax function, may not be used?
    """
    x = x - np.max(x)
    exp_x = np.exp(x * t)
    return exp_x / np.sum(exp_x)


def generate_weighted_distribution(loss):
    """
    generate a weight from the loss.
    :param loss:
    :return: weights
    """
    loss = [(1 / each) for each in loss]
    loss = softmax(loss)
    return loss


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))