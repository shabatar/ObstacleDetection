import cv2
import numpy as np
import math

class Geometry:
    def moduleVect3D(self, vector):
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    @staticmethod
    def unitVect(vector):
        return vector / np.linalg.norm(vector)
    @staticmethod
    def angleVect(v1, v2):
        v1_u = Geometry.unitVect(v1)
        v2_u = Geometry.unitVect(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    @staticmethod
    def dist3D(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    @staticmethod
    def distToLine(line, point):
        myLine = (line[0], line[1], line[2])
        myPoint = (point[0], point[1])
        dist = abs(line[0] * point[0] + line[1] * point[1] + line[2]) / math.sqrt(line[0] * line[0] + line[1] * line[1])
        return dist
    @staticmethod
    def dist(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    @staticmethod
    def moduleVect(vector):
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    @staticmethod
    def planeByPoints(pt1, pt2, pt3):
        x1,y1,z1 = pt1[0], pt1[1], pt1[2]
        x2,y2,z2 = pt2[0], pt2[1], pt2[2]
        x3,y3,z3 = pt3[0], pt3[1], pt3[2]
        # Ax + By + Сz + D = 0
        # det{ {x-x1, y-y1, z-z1}, {x2-x1, y2-y1, z2-z1}, {x3-x1, y3-y1, z3-z1} } = 0
        # x2-x1 = a2, y2-y1 = b2, z2-z1 = c2 etc.
        A = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
        B = (z2-z1)*(x3-x1)-(z3-z1)*(x2-x1)
        C = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
        D = x1*(-A)+y1*(-B)+z1*(-C)
        return A, B, C, D
    @staticmethod
    def distPlanePt(plane, pt):
        plane = np.array(plane)
        pt = np.array(pt)
        pt = pt.tolist()
        plane = plane.tolist()
        #if (type(pt[0]) is list):
        #    pt = [i[0] for i in plane]
        ptx, pty, ptz = pt[0], pt[1], pt[2]
        A, B, C, D = plane[0], plane[1], plane[2], plane[3]
        if (A == 0.0 and B == 0.0 and C == 0.0):
            return 10000
        up = abs(A*ptx + B*pty + C*ptz + D)
        down = math.sqrt(A**2 + B**2 + C**2)
        dst = up / down
        return dst


class VisOdometry:
    #ptsold = pts1, ptsnew = pts2
    def __init__(self, cam, ptsold, ptsnew):
        self.F, self.fundamentalmask = cv2.findFundamentalMat(ptsold, ptsnew, cv2.FM_8POINT)
        if (self.F is not None):
            # Find epilines corresponding to points in second image
            self.epilinesold = cv2.computeCorrespondEpilines(ptsnew.reshape(-1, 1, 2), 2, self.F)
            self.epilinesold = self.epilinesold.reshape(-1, 3)
            # Find epilines corresponding to points in first image
            self.epilinesnew = cv2.computeCorrespondEpilines(ptsold.reshape(-1, 1, 2), 2, self.F)
            self.epilinesnew = self.epilinesnew.reshape(-1, 3)
        else:
            self.F, self.fundamentalmask = cv2.findFundamentalMat(ptsold, ptsnew, cv2.LMEDS)
            if (self.F is not None):
                # Find epilines corresponding to points in second image
                self.epilinesold = cv2.computeCorrespondEpilines(ptsnew.reshape(-1, 1, 2), 2, self.F)
                self.epilinesold = self.epilinesold.reshape(-1, 3)
                # Find epilines corresponding to points in first image
                self.epilinesnew = cv2.computeCorrespondEpilines(ptsold.reshape(-1, 1, 2), 2, self.F)
                self.epilinesnew = self.epilinesnew.reshape(-1, 3)
            else:
                print('F not estimated')
        # ptsold, ptsnew ?
        self.E, self.essentialmask = cv2.findEssentialMat(ptsold, ptsnew, focal=cam.focal, pp=cam.pp, method=cv2.RANSAC,
                                                 prob=0.999, threshold=1.0)
        # K'FK
        self.E = np.dot(np.dot(cam.calibMat.transpose(), self.F),cam.calibMat)
        # essentialmask ?
        if (self.E is None):
            print("E is None")
            return
        if (self.essentialmask is not None):
            # detect outliers and inliers, pts = inliers
            self.outold = ptsold[self.essentialmask.ravel() == 0]
            self.outnew = ptsnew[self.essentialmask.ravel() == 0]
            ptsold = ptsold[self.essentialmask.ravel() == 1]
            ptsnew = ptsnew[self.essentialmask.ravel() == 1]
        _, self.R, self.t, _ = cv2.recoverPose(self.E, ptsold, ptsnew, focal=cam.focal, pp=cam.pp)