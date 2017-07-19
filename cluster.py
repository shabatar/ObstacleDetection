import cv2
import numpy as np
import math


def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

class Cluster:
    def getNearbyPoints(self, unassigned, point, eps):
        if point in unassigned:
            unassigned.remove(point)
        nearby = [point]
        while (len(unassigned) > 0):
            nextNeighbours = []
            for neighbour in nearby:
                for pretender in unassigned:
                    if 0.001 < dist(neighbour, pretender) < eps:
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
