import cv2
import numpy as np
import math
from visodometry import Geometry

class Cluster:
    #def __init__(self, oldpts, newpts):
    #    self.oldpts = oldpts
    #    self.newpts = newpts

    def getNearbyPoints(self, unassigned, point, eps):
        if point in unassigned:
            unassigned.remove(point)
        nearby = [point]
        while (len(unassigned) > 0):
            nextNeighbours = []
            for neighbour in nearby:
                for pretender in unassigned:
                    if 0.001 < Geometry.dist(neighbour, pretender) < eps:
                        nextNeighbours.append(pretender)
                        unassigned.remove(pretender)
            nearby += nextNeighbours
            if len(nextNeighbours) == 0:
                return nearby
                # map(lambda p: unassigned.remove(p), nearby)
        return nearby

    def clusterPoints(self, points, eps):  # eps = num of pixels to be considered as close enough
        unassigned = [(point[0], point[1]) for point in points]
        clusters = []
        while (len(unassigned)):
            for p in unassigned:
                myPoint = (p[0], p[1])
                # eps=52 is a good number
                near = self.getNearbyPoints(unassigned, myPoint, eps)
                clusters.append(near)
        return clusters
