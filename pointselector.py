import cv2
import numpy as np
import random
from drawer import Drawer
import math
from constants import MagicConstants
from visodometry import Geometry
from drawer import Drawer

class OpticalFlow:
    def __init__(self, img1, img2, read, blur):
        if (not read):
            self.img1 = cv2.imread(img1)
            #print(self.img1.shape)
            self.img2 = cv2.imread(img2)
        else:
            self.img1 = img1
            self.img2 = img2
        if (blur):
            self.img1 = cv2.blur(self.img1, MagicConstants.gaussWin)
            self.img2 = cv2.blur(self.img2, MagicConstants.gaussWin)
        # initial points
        self.pts1 = []
        self.pts2 = []
        self.drw = Drawer(self.img1, self.img2)

    def cornerHarris(self):
        img = self.img2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 0.04
        dst = cv2.cornerHarris(gray, 4, 3, 0.09)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        res = np.hstack((centroids, corners))
        res = np.int0(res)
        try:
            img[res[:, 1], res[:, 0]] = [0, 0, 255]
            img[res[:, 3], res[:, 2]] = [0, 255, 0]
        except Exception:
            print("kek")
            corners = corners[:, np.newaxis]
            return corners

        #cv2.imwrite('harris.png',img)
        corners = corners[:, np.newaxis]
        corners1 = []
        for p in corners:
            p1 = [p[0][0], p[0][1]]
            corners1.append(p1)
        #self.drw.drawPoints(corners1, 'harris.png', self.img1, True)
        return corners

    def lucasKanade(self):
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        '''
        img_yuv = cv2.cvtColor(self.img1, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.img1 = img_output

        img_yuv = cv2.cvtColor(self.img2, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.img2 = img_output
        '''
        '''
        old_frame = self.img1
        frame = self.img2
        size = self.img1.shape
        newh = size[0] // 3
        old_frame = old_frame[newh: size[0], 1: size[1]]
        frame = frame[newh: size[0], 1: size[1]]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''
        oldFrame = self.img1
        newFrame = self.img2
        oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
        frameGray = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('kek.png', oldFrame)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        p0 = self.cornerHarris()

        mask = np.zeros_like(oldFrame)
        p1, st, err = cv2.calcOpticalFlowPyrLK(oldGray, frameGray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        #self.drw.drawFlow(good_old, good_new, 'optFlow.png', self.img1, True)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), [0, 0, 255], 2)
            newFrame = cv2.circle(newFrame, (a, b), 2, [0, 0, 255], -1)
        img = cv2.add(newFrame, mask)
        # cv2.imwrite('opticalFlow.jpg', img)
        self.pts1 = good_old
        self.pts2 = good_new
        return good_new, good_old

class PointSelector:
    def __init__(self, img1, img2, read=False, blur=True):
        self.optFlow = OpticalFlow(img1, img2, read, blur)

    def largeModule(self, vector):
        return (Geometry.moduleVect(vector) > MagicConstants.largeModuleCnst * self.optFlow.img1.shape[0] or
                Geometry.moduleVect(vector) > MagicConstants.largeModuleCnst * self.optFlow.img1.shape[1])

    def getPtsOnRoad(self):
        height, width, depth = self.optFlow.img2.shape
        newh = height // 10
        neww = width // 9
        img1 = cv2.cvtColor(self.optFlow.img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.optFlow.img2, cv2.COLOR_BGR2GRAY)
        new0y = newh * 8
        new0x = neww * 3
        self.bot1 = img1[new0y: newh * 10, new0x: 4 * neww]
        self.bot2 = img2[new0y: newh * 10, new0x: 4 * neww]
        flow = cv2.calcOpticalFlowFarneback(self.bot1, self.bot2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pts1 = []
        pts2 = []
        for i in range(0, self.bot1.shape[1], 4):
            for j in range(0, self.bot1.shape[0], 4):
                #good version
                point2 = [new0x + i + 0.0, new0y + j + 0.0]
                point1 = [new0x + i - flow[j][i][0] + 0.0, new0y + j - flow[j][i][1] + 0.0]
                #point1 = [3 * neww + i + 0.0, 3 * newh + j + 0.0]
                #point2 = [3 * neww + i + flow[j][i][0] + 0.0, 3 * newh + j + flow[j][i][1] + 0.0]
                pts1.append(point1)
                pts2.append(point2)
        #randIndex = random.sample(range(len(pts1)), MagicConstants.pointToSelect)
        #randIndex.sort()
        #pts1 = [pts1[i] for i in randIndex]
        #pts2 = [pts2[i] for i in randIndex]
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        return pts1, pts2

    def select(self):
        # imgs, good_old_points in imgs, good_new_points in imgs
        # returns good_enough_old_p and new
        good_new, good_old = self.optFlow.lucasKanade()

        good_new1 = [[point[0], point[1]] for point in good_new]
        good_old1 = [[point[0], point[1]] for point in good_old]
        #remove too large vectors
        for new, old in zip(good_new1, good_old1):
            vx = old[0] - new[0]
            vy = old[1] - new[1]
            v = [vx, vy]
            if (self.largeModule(v)):
                good_new1.remove(new)
                good_old1.remove(old)
        goodnum = MagicConstants.pointToSelect
        img1 = self.optFlow.img1
        img2 = self.optFlow.img2
        size = img2.shape
        newh = size[0] // 3
        self.top_img1 = img1[1: newh, 1: size[1]]
        self.mid_img1 = img1[newh: newh * 2, 1: size[1]]
        self.bot_img1 = img1[newh * 2: size[0], 1: size[1]]
        self.top_img2 = img2[1: newh, 1: size[1]]
        self.mid_img2 = img2[newh: newh * 2, 1: size[1]]
        self.bot_img2 = img2[newh * 2: size[0], 1: size[1]]
        pts = {}
        t0 = []
        m0 = []
        b0 = []
        topimgcnt = 0
        midimgcnt = 0
        botimgcnt = 0
        chimgcnt = 0
        allcnt = 0
        mask = np.zeros_like(img1)
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new[0], new[1]  # good new x, y
            c, d = old[0], old[1]  # good old x, y
            allcnt += 1
            if b < newh and d < newh:
                topimgcnt += 1
                t0.append((new, old))
                # pts[(new, old)] = "top"
            elif b < newh * 2 and d < newh * 2:
                midimgcnt += 1
                m0.append((new, old))
                # pts[(new, old)] = "mid"
            elif b < size[0] and d < size[0]:
                botimgcnt += 1
                b0.append((new, old))
                # pts[(nexrw, old)] = "road"
            else:
                chimgcnt += 1
        goodnum1 = min(goodnum, min(len(b0), min(len(m0), len(t0))))
        #if (abs(goodnum1 - goodnum) > MagicConstants.subtractGNum):
        #    road = b0
        #    road_new, road_old = zip(*road)
        #    return good_new, good_old, road_new, road_old
        goodnum = goodnum1
        if (goodnum == 0):
            road_new, road_old = self.getPtsOnRoad()
            return good_new, good_old, road_new, road_old
        road = random.sample(b0, goodnum)
        t0 = random.sample(m0, goodnum) + road
        '''random.sample(t0, goodnum) + '''
        for i, (new, old) in enumerate(t0):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), [30, 255, 255], 2)
            img2 = cv2.circle(img2, (a, b), 2, [30, 255, 255], -1)
        img = cv2.add(img2, mask)
        # print(road)
        cv2.imwrite('goodEnough.jpg', img)
        if(t0):
            good_new, good_old = zip(*t0)
        #comment above to do nothing
        if (len(road)!=0):
            road_new, road_old = zip(*road)
        else:
            road_new, road_old = self.getPtsOnRoad()
        #if (not good_new or not good_old):
        #    return (0,0)
        # было просто
        good_old = np.int32(good_old)
        good_new = np.int32(good_new)

        road_new = np.int32(road_new)
        road_old = np.int32(road_old)
        return good_new, good_old, road_new, road_old