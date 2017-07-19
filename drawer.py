import cv2
import numpy as np

class Drawer:
    def __init__(self, img1, img2, color=[255,0,0]):
        self.img1 = img1
        self.img2 = img2
        self.color = color

    def drawEpilines(self, img1, img2, lines, pts1, pts2):
        img1 = cv2.imread(img1)  # draw epilines here
        img2 = cv2.imread(img2)
        r, c = img1.shape[:2]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # color = [0,0,255]
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 2, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 2, color, -1)
        return img1, img2

    def drawPoints(self, pts1, imgout, img, read=False):
        if (not read):
            frame = cv2.imread(img)
        else:
            frame = img
        for p in pts1:
            #print(p)
            a, b = p[0], p[1]
            frame = cv2.circle(frame, (a, b), 2, [0, 0, 255], -1)
        cv2.imwrite(imgout, frame)
        return frame

    def drawFlow(self, pts1, pts2, imgout, img, read=False):
        if (not read):
            frame = cv2.imread(img)
        else:
            frame = img
        mask = np.zeros_like(frame)

        for i, (new, old) in enumerate(zip(pts1, pts2)):
            a, b = new[0], new[1]
            c, d = old[0], old[1]
            mask = cv2.line(mask, (a, b), (c, d), [0, 0, 255], 2)
            frame = cv2.circle(frame, (a, b), 2, [0, 0, 255], -1)
        img = cv2.add(frame, mask)
        cv2.imwrite(imgout, img)
        return img

    def drawClusters(self, clusterarrs, imgsrc, color=None):
        color = color or self.color
        frame1 = cv2.imread(imgsrc)
        for cluster in clusterarrs:
            maxX = max(cluster, key=lambda p: p[0])[0]
            maxY = max(cluster, key=lambda p: p[1])[1]
            minX = min(cluster, key=lambda p: p[0])[0]
            minY = min(cluster, key=lambda p: p[1])[1]
            for p in cluster:
                a, b = p
                cv2.circle(frame1, (a, b), 4, color, -1)
                cv2.rectangle(frame1, (minX, maxY), (maxX, minY), color, 3)
        return frame1

    def drawRects(self, rects, imgsrc, color=None):
        color = color or self.color
        frame1 = cv2.imread(imgsrc)
        for rect in rects:
            minX, maxY, maxX, minY = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
            #print(minY)
            cv2.rectangle(frame1, (minX, maxY), (maxX, minY), color, 3)
        #cv2.imwrite('kek.png',frame1)
        return frame1