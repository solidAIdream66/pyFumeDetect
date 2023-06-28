"""Detect lane by hough line transform.
"""

import streamlit as st
import cv2
from trackTruck import playVideo
import numpy as np


class LineEx:
    def __init__(self, line, slope, const):
        self.line = line
        self.slope = slope
        self.const = const


def regionOfInterest(img, vertics):
    """Select the region of interest (ROI) from a defined list of vertices."""
    mask = np.zeros_like(img)

    # Defining color to fill the mask
    if len(img.shape) > 2:
        channel_count = img.shape(2)
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon
    cv2.fillPoly(mask, vertics, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero.
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """Utility for drawing lines."""
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def calc_line_parameters(lines):
    lines_ex = []
    if (lines is not None) and (len(lines) != 0):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 - x2 != 0:
                    slope = (y1 - y2) / (x1 - x2)
                else:
                    slope = (y1 - y2) / 0.1
                c = y1 - slope * x1
                line_ex = LineEx([x1, y1, x2, y2], slope, c)
                lines_ex.append(line_ex)

    return lines_ex


def filter_slope_range(lines_ex, upper_slope, lower_slope):
    linesEx_filtered = []

    for lineEx in lines_ex:
        if lineEx.slope < upper_slope or lineEx.slope > lower_slope:
            linesEx_filtered.append(lineEx)
    return linesEx_filtered


def cluster_lines(lines_ex, slope_gap, const_gap):
    clusters = []
    for i_line, lineEx in enumerate(lines_ex):
        is_cluster = False
        for cluster in clusters:
            for i_cluster in cluster:
                if abs(lineEx.slope - lines_ex[i_cluster].slope) < slope_gap and \
                        abs(lineEx.const - lines_ex[i_cluster].const) < const_gap:
                    cluster.append(i_line)
                    is_cluster = True
                    break

            if is_cluster:
                break
        if not is_cluster:
            clusters.append([i_line])
        clusters.sort(key=lambda x: len(x), reverse=True)

    return clusters


def cal_avg(values):
    """Calculate average value."""
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 1
        return sum(values) / n


def extrapolated_lane_image(clusters, lines_ex, upper_border, lower_border, line_number_threshold):
    """Main function called to get the final lane lines"""
    lines_combined = []
    for cluster in clusters:
        if len(cluster) <= line_number_threshold:
            break
        slopes = []
        consts = []
        for i_cluster in cluster:
            slopes.append(lines_ex[i_cluster].slope)
            consts.append(lines_ex[i_cluster].const)

        avg_slope = cal_avg(slopes)
        avg_consts = cal_avg(consts)
        x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)
        x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)

        lines_combined.append([x_lane_lower_point, lower_border, x_lane_upper_point, upper_border])
    return lines_combined


def laneDetect(frame):
    # 1. Create Threshold for Lane Lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_select = cv2.inRange(gray, 220, 255)

    # 2. Select Region of Interest
    roi_vertics = np.array([[[850, 100], [1600, 100], [2039, 436], [2039, 1530], [200, 1530]]])
    gray_select_roi = regionOfInterest(gray_select, roi_vertics)

    # 3. Detect Edges using Canny Edge Detector
    low_threshold = 100
    high_threshold = 200
    img_canny = cv2.Canny(gray_select_roi, low_threshold, high_threshold)

    # 3.1 Remove noise using Gaussian blur
    kernal_size = 11
    canny_blur = cv2.GaussianBlur(img_canny, (kernal_size, kernal_size), 0)

    # 4. Fit lines using Hough Line Transform
    rho = 1
    theta = np.pi / 180
    threshold = 100
    min_line_len = 300
    max_line_gap = 5
    lines = hough_lines(canny_blur, rho, theta, threshold, min_line_len, max_line_gap)

    # 4.1 calculate slope, consts of lines
    lines_ex = calc_line_parameters(lines)
    # print(len(lines_ex))

    # 4.2 filter the lines by slope range
    left_slope = (109 - 1530) / (886 - 256) + 0.2
    right_slope = (94 - 436) / (1560 - 2039) - 0.2
    lines_ex = filter_slope_range(lines_ex, left_slope, right_slope)
    # print(len(lines_ex))

    # 4.3 cluster by slope and consts, sort by number
    slope_gap = 0.1
    const_gap = 50
    clusters = cluster_lines(lines_ex, slope_gap, const_gap)
    # print(clusters)

    # 5. Extrapolate the lanes from lines found
    roi_upper_border = 100
    roi_lower_border = 1530
    line_number_threshold = 1
    lines = extrapolated_lane_image(clusters, lines_ex, roi_upper_border, roi_lower_border, line_number_threshold)
    # print(lines)

    # Composite the result with original image
    lane_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    draw_lines(lane_img, lines)
    image_result = cv2.addWeighted(frame, 1, lane_img, 2, 0.0)
    return image_result


def clickEvent(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)


def showFrame(frame):
    cv2.imshow(win_name, frame)

    # temporally to get coordinates of Region of Interest
    if False:
        cv2.setMouseCallback(win_name, clickEvent)

    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        return False
    return True


if __name__ == '__main__':
    win_name = 'Lane detection using hough line transform'
    cv2.namedWindow(win_name)

    playVideo('Traffic Videos/test.mp4', laneDetect, showFrame)

    cv2.destroyWindow(win_name)
