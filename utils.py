import numpy as np
import cv2

def order_points(points):
    """
    this function returns the points in the following order:
    top_left, top_right, bottom_right, bottom_left
    """
    new_points = np.zeros((4, 2), dtype="float32")
    summed = points.sum(axis=1)
    new_points[0] = points[np.argmin(summed)]
    new_points[2] = points[np.argmax(summed)]

    diffed = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diffed)]
    new_points[3] = points[np.argmax(diffed)]

    return np.float32(new_points)


def order_contours(cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][0], reverse=True))
    return cnts