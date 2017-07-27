import cv2
import numpy as np
import random
import math
from constants import MagicConstants
from visodometry import Geometry

class Outlierer:
    #def __init__(self):
    #    self.outEss = outEss

    def selectOutliers(self, points, epilines):
        outliers = []
        for (p, line) in zip(points, epilines):
            if (Geometry.distToLine(line, p) > MagicConstants.closeToEpipolar):
                outliers.append(p)
        return outliers

    def optFLowMagnOutliers(self, pts1, pts2):
        out1 = []
        out2 = []
        pts1 = [(point[0], point[1]) for point in pts1]
        pts2 = [(point[0], point[1]) for point in pts2]
        for (p1, p2) in zip(pts1, pts2):
            if (Geometry.dist(p1, p2) < MagicConstants.optFlowMagn):
                out1.append(p1)
                out2.append(p2)
        return out1, out2
