import cv2
import numpy as np


if __name__ == "__main__":
    h_video = cv2.VideoCapture('test.mp4')
    h_bg = cv2.createBackgroundSubtractorKNN(history = 200)
    win_name = 'Traffic Video with Fume Preview'
    ksize_erode = (11, 11)
    ksize_dilate = (101, 101)
    red = (0, 0, 255)
    yellow = (0, 255, 255)
    truck_area_threshold = 100000

    cv2.namedWindow(win_name)

    while True:
        is_frame, frame = h_video.read()
        if not is_frame:
            break

        frame_erode = frame.copy()
        fg_mask = h_bg.apply(frame)
        fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize_erode, np.uint8))
        fg_mask_dilate = cv2.dilate(fg_mask_erode, np.ones(ksize_dilate, np.uint8))

        frame_fg_mask_erode = cv2.cvtColor(fg_mask_dilate, cv2.COLOR_GRAY2BGR)
        contours_erode, hierarchy = cv2.findContours(fg_mask_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours_erode) > 0:
            contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)

            for i in range(len(contours_sorted)):
                # print(cv2.contourArea(contours_sorted[i]))
                if cv2.contourArea(contours_sorted[i]) < truck_area_threshold:
                    continue

                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[i])

                cv2.rectangle(frame_erode, (xc, yc), (xc+wc, yc+hc), yellow, thickness=2)
                cv2.rectangle(frame_fg_mask_erode, (xc, yc), (xc+wc, yc+hc), yellow, thickness=2)

        frame_composite = np.hstack([frame_fg_mask_erode, frame_erode])
        cv2.imshow(win_name, frame_composite)

        key = cv2.waitKey(1)

        if key == ord('Q') or key == ord('q') or key == 27:
            break

    h_video.release()
    cv2.destroyWindow(win_name)
