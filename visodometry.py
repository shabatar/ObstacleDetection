import cv2
import numpy as np
import math

class Geometry:
    def dist(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    def moduleVect(self, vector):
        return math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    def unitVect(self, vector):
        return vector / np.linalg.norm(vector)
    def angleVect(self, v1, v2):
        v1_u = self.unitVect(v1)
        v2_u = self.unitVect(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    def dist3D(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


class VisOdometry:
    #ptsold = pts1, ptsnew = pts2
    def __init__(self, cam, ptsold, ptsnew):
        # focal = fx
        self.E, self.essentialmask = cv2.findEssentialMat(ptsold, ptsnew, focal=cam.focal, pp=cam.pp, method=cv2.RANSAC,
                                                 prob=0.999, threshold=1.0)
        #out2 = pts2[mask.ravel() == 0]
        if (self.E is None):
            print("tresh")
            return
        if (self.essentialmask is not None):
            # detect outliers and inliers, pts = inliers
            self.outold = ptsold[self.essentialmask.ravel() == 0]
            self.outnew = ptsnew[self.essentialmask.ravel() == 0]
            ptsold = ptsold[self.essentialmask.ravel() == 1]
            ptsnew = ptsnew[self.essentialmask.ravel() == 1]
        _, self.R, self.t, mask = cv2.recoverPose(self.E, ptsold, ptsnew, focal=cam.focal, pp=cam.pp)
        self.F, self.fundamentalmask = cv2.findFundamentalMat(ptsold, ptsnew, cv2.FM_8POINT)
        if (self.F is not None):
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            self.epilinesold = cv2.computeCorrespondEpilines(ptsnew.reshape(-1, 1, 2), 2, self.F)
            self.epilinesold = self.epilinesold.reshape(-1, 3)
            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            self.epilinesnew = cv2.computeCorrespondEpilines(ptsold.reshape(-1, 1, 2), 2, self.F)
            self.epilinesnew = self.epilinesnew.reshape(-1, 3)
        else:
            self.F, self.fundamentalmask = cv2.findFundamentalMat(ptsold, ptsnew, cv2.LMEDS)
            if (self.F is not None):
                # Find epilines corresponding to points in right image (second image) and
                # drawing its lines on left image
                self.epilinesold = cv2.computeCorrespondEpilines(ptsnew.reshape(-1, 1, 2), 2, self.F)
                self.epilinesold = self.epilinesold.reshape(-1, 3)
                # Find epilines corresponding to points in left image (first image) and
                # drawing its lines on right image
                self.epilinesnew = cv2.computeCorrespondEpilines(ptsold.reshape(-1, 1, 2), 2, self.F)
                self.epilinesnew = self.epilinesnew.reshape(-1, 3)
            else:
                print('dich')