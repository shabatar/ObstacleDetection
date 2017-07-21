import cv2
import numpy as np

class Tracker:
    def trackRect(self, rect, frame):
        # take first frame of the video
        frame = cv2.imread(frame)
        # setup initial location of window
        minX, maxY, maxX, minY = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
        c = minX
        r = minY
        w = maxX - minX
        h = maxY - minY
        track_window = (c, r, w, h)
        if (c < 0 or r < 0 or w < 0 or h < 0):
            return [(1, 2), (3, 4)]
        if (minX == maxX or minY == maxY):
            return [(1, 2), (3, 4)]
        # set up the ROI for tracking
        roi = frame[int(r):int(r + h), int(c):int(c + w)]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        # (minX, maxY), (maxX, minY)
        img2 = cv2.rectangle(frame, (x, y + h), (x + w, y), 255, 2)
        # cv2.imwrite('img2.png', img2)
        rect = [(x, y + h), (x + w, y)]
        return rect

    def trackRects(self, rectas, img):
        rects = []
        for rect in rectas:
            rect1 = self.trackRect(rect, img)
            rects.append(rect1)
        return rects