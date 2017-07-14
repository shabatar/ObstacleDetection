import cv2
import numpy as np
import random
import math
from constants import MagicConstants

cnst = MagicConstants()

class Outlierer:
    #def __init__(self):
    #    self.outEss = outEss

    def dist(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def distToLine(self, line, point):
        myLine = (line[0], line[1], line[2])
        myPoint = (point[0], point[1])
        dist = abs(line[0] * point[0] + line[1] * point[1] + line[2]) / math.sqrt(line[0] * line[0] + line[1] * line[1])
        return dist

    def selectOutliers(self, points, epilines):
        outliers = []
        for (p, line) in zip(points, epilines):
            if (self.distToLine(line, p) > cnst.closeToEpipolar):  # ЗАХАРДКОДИМ?
                outliers.append(p)
        return outliers

    def optFLowMagnOutliers(self, pts1, pts2):
        out1 = []
        out2 = []
        pts1 = [(point[0], point[1]) for point in pts1]
        pts2 = [(point[0], point[1]) for point in pts2]
        for (p1, p2) in zip(pts1, pts2):
            if (self.dist(p1, p2) < cnst.optFlowMagn):  # ЗАХАРДКОДИМ?
                out1.append(p1)
                out2.append(p2)
        return out1, out2
