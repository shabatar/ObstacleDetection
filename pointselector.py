import cv2
import numpy as np
import random
from drawer import Drawer
from constants import MagicConstants

cnst = MagicConstants()

class PointSelector:
    def __init__(self, img1, img2):
        self.img1 = cv2.imread(img1)
        self.img1 = cv2.blur(self.img1, cnst.gaussWin)
        #self.img1 = cv2.pyrMeanShiftFiltering(self.img1, cnst.meanShiftN, cnst.meanShiftN)
        self.img2 = cv2.imread(img2)
        self.img2 = cv2.blur(self.img2, cnst.gaussWin)
        #self.img2 = cv2.pyrMeanShiftFiltering(self.img2, cnst.meanShiftN, cnst.meanShiftN)
        self.magicConstant = cnst.pointToSelect

    def cornerHarris(self):
        img = self.img1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 0.04
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        res = np.hstack((centroids, corners))
        res = np.int0(res)
        img[res[:, 1], res[:, 0]] = [0, 0, 255]
        img[res[:, 3], res[:, 2]] = [0, 255, 0]
        # cv2.imwrite('harris.png',img)
        corners = corners[:, np.newaxis]
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
        old_frame = self.img1
        frame = self.img2
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('kek.png', old_frame)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        p0 = self.cornerHarris()

        mask = np.zeros_like(old_frame)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), [0, 0, 255], 2)
            frame = cv2.circle(frame, (a, b), 2, [0, 0, 255], -1)
        img = cv2.add(frame, mask)
        # cv2.imwrite('opticalFlow.jpg', img)
        return good_new, good_old

    def select(self):
        # imgs, good_old_points in imgs, good_new_points in imgs
        # returns good_enough_old_p and new
        good_new, good_old = self.lucasKanade()
        #drawer
        goodnum = self.magicConstant  # ЗАХАРДКОДИМ? 42
        img1 = self.img1
        img2 = self.img2
        size = img2.shape
        newh = size[0] // 3
        top_img2 = img2[1: newh, 1: size[1]]
        mid_img2 = img2[newh: newh * 2, 1: size[1]]
        bot_img2 = img2[newh * 2: size[0], 1: size[1]]
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
            a, b = new.ravel()  # good new x, y
            c, d = old.ravel()  # good old x, y
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
                # pts[(new, old)] = "road"
            else:
                chimgcnt += 1
        goodnum = min(goodnum, min(len(b0), min(len(m0), len(t0))))
        if (goodnum == 0):
            t0 = t0 + m0 + b0
            good_new, good_old = zip(*t0)
            return good_new, good_old
        # goodnum2 = min(60, len(m0))
        road = random.sample(b0, goodnum)
        t0 = random.sample(t0, goodnum) + random.sample(m0, goodnum) + road
        for i, (new, old) in enumerate(t0):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), [255, 0, 0], 2)
            img2 = cv2.circle(img2, (a, b), 2, [255, 0, 0], -1)
        img = cv2.add(img2, mask)
        cv2.imwrite('goodEnough.jpg', img)
        good_new, good_old = zip(*t0)
        #comment above to do nothing

        road_new, road_old = zip(*road)
        #if (not good_new or not good_old):
        #    return (0,0)
        return good_new, good_old, road_new, road_old