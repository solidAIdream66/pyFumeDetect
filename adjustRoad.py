"""adjust perspective of road using Homography transform
"""

import cv2
from trackTruck import playVideo, ShowFrame
import numpy as np


def adjustRoad(frame):
    # 1. 4 keypoints before and after
    # points_1 = np.array([[752, 328], [181, 1357], [1417, 465], [1691, 1024]], dtype=np.float32)
    # points_2 = np.array([[181, 328], [181, 1357], [1691, 465], [1691, 1024]], dtype=np.float32)
    points_1 = np.array([[1449, 6], [1982, 382], [1200, 31], [1568, 833]], dtype=np.float32)
    points_2 = np.array([[1982, 6], [1982, 382], [1200, 31], [1200, 833]], dtype=np.float32)

    # 2. Computing the homography that relates the two images
    h, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC)

    # 3. Using the homography to warp the perspective of the original image
    height, width, channels = frame.shape
    frame_adjusted = cv2.warpPerspective(frame, h, (width, height))
    return frame_adjusted


def clickEvent(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)


if __name__ == '__main__':
    win_name = 'adjust perspective of road using Homography transform'
    cv2.namedWindow(win_name)

    # temporally to get coordinates of Region of Interest
    if False:
        cv2.setMouseCallback(win_name, clickEvent)

    playVideo('Traffic Videos/test.mp4', adjustRoad, ShowFrame(win_name))

    cv2.destroyWindow(win_name)
