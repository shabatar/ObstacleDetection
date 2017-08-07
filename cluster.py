import cv2
import numpy as np
import math
from visodometry import Geometry
from constants import MagicConstants

class Cluster:
    def __init__(self, oldpts, newpts):
        self.clstoldpts = self.clusterPoints(oldpts, MagicConstants.clusterConstant)
        self.clstnewpts = self.clusterPoints(newpts, MagicConstants.clusterConstant)
        clVect = self.clusterVect(clustersold=self.clstoldpts, clustersnew=self.clstnewpts)
        clVect = self.removeOutliers(clVect)
        self.clstnewpts, self.clstoldpts = self.recoverClusters(self.clstnewpts, clVect)

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

    def clusterVect(self, clustersold, clustersnew):
        clusters = []
        for clust1, clust2 in zip(clustersold, clustersnew):
            clust = []
            for p1, p2 in zip(clust1, clust2):
                v = [p1[0] - p2[0], p1[1] - p2[1]]
                clust.append(v)
            clusters.append(clust)
        return clusters

    def removeOutliers(self, clustervects):
        outclusters = []
        for cluster in clustervects:
            cnt = 0
            for v1 in cluster:
                for v2 in cluster:
                    alpha = Geometry.angleVect(np.array(v1), np.array(v2))
                    if (alpha < MagicConstants.critAlpha):
                        cnt += 1
            if (cnt <= (len(cluster) ** 2) // 4):  # bad cluster
                outclusters.append([0])
                continue
            else:
                outclusters.append(cluster)
        return outclusters

    def recoverClusters(self, newclust, vectclust):  # recover clusters without outliers
        newclusters = []
        oldclusters = []
        for clust1, clust2 in zip(newclust, vectclust):
            if (len(clust2) == 1):  # it's a point, bro! old clusters should not be recovered
                continue
            else:
                oldclust = []
                for p1, p2 in zip(clust1, clust2):
                    newp = [p2[0] + p1[0], p2[1] + p1[1]]
                    oldclust.append(newp)
                oldclusters.append(oldclust)
                newclusters.append(clust1)
        return newclusters, oldclusters

    def minDistToCluster(self, clust1, clust2, reconst):
        minDist = 1e6
        for p1, p2 in zip(clust1, clust2):
            # print("mindist")
            p3d = reconst.reconstructPoint(p1, p2)
            if p3d[2] < minDist:
                minDist = abs(p3d[2])
        return minDist

    def minDistToClusters(self, clusts1, clusts2, reconst, scaleFactor):
        d = []
        for clust1, clust2 in zip(clusts1, clusts2):
            di = self.minDistToCluster(clust1, clust2, reconst)
            if (di != 1e6):
                print(di)
                print(di * scaleFactor / 100)
                d.append(di * scaleFactor / 100)
        return d
