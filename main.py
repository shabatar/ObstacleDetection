import cv2
import numpy as np
import random
import math
from matplotlib import pyplot as plt
import matplotlib
from numpy import linalg
from pointselector import PointSelector
from drawer import Drawer
#from clusterizer import Clusterizer
from outlierer import Outlierer
from camera import Camera
from constants import MagicConstants
from tracker import Tracker
from reconstructor import Reconstructor
from cluster import Cluster
from visodometry import VisOdometry
from visodometry import Geometry

# load calibration data from file

with open('calib.txt') as f:
    f = f.readlines()

for line in f:
    tokens = line.split()
    if (len(tokens) == 0):
        continue
    if (tokens[0] == 'S_02:'):
        size = (float(tokens[1]), float(tokens[2]))
    elif (tokens[0] == 'K_02:'):
        fx, px, fy, py = float(tokens[1]), float(tokens[3]), float(tokens[5]), float(tokens[6])
        calibMat = np.array([
            [fx,  0.0,  px],
            [0.0,  fy,  py],
            [0.0, 0.0, 1.0]
        ])
        focal = fx
        pp = (px, py)
    elif (tokens[0] == 'D_02:'):
        distCoeffs = [float(tok) for tok in tokens[1:]]
    else:
        continue

tracker = Tracker()
route = "data/00000000*.png"
# class-helper
geom = Geometry()

first_two_frames = False

# If need to show detected
show = True
# If need to use road reconstruction
reconstruct = False
# If need to use reconstructed plotting
plot = False

cam = Camera(calibMat, distCoeffs, focal, pp, size)

def inRect(point, rect):
    minX, maxY, maxX, minY = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    return (point[0] < maxX and point[0] > minX and point[1] < maxY and point[1] > minY)

def closeRects(rect1, rect2):
    minX, maxY, maxX, minY = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
    r1p1 = [minX, minY]
    r1p2 = [maxX, minY]
    r1p3 = [minX, maxY]
    r1p4 = [maxX, maxY]
    return inRect(r1p1, rect2) or inRect(r1p2, rect2) or inRect(r1p3, rect2) or inRect(r1p4, rect2)

def largeEnough(rect):
    minX, maxY, maxX, minY = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    S = (maxX - minX) * (maxY - minY)
    return (S > 50)

def unionRects(rect1, rect2):
    minX1, maxY1, maxX1, minY1 = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
    minX2, maxY2, maxX2, minY2 = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]
    return [[min(minX1, minX2), max(maxY1, maxY2)], [max(maxX1, maxX2), min(minY1, minY2)]]

def unionCloseRects(rects1, rects2):
    union = []
    for rect1 in rects1:
        for rect2 in rects2:
            if rect1[0] == rect2[0]:
                continue
            minX, maxY, maxX, minY = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
            d1 = math.sqrt((maxX-minX) ** 2 + (maxY-minY) ** 2)
            c1 = (abs(maxX - minX) / 2, abs(maxY - minY) / 2)
            minX, maxY, maxX, minY = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]
            d2 = math.sqrt((maxX-minX) ** 2 + (maxY-minY) ** 2)
            c2 = (abs(maxX - minX) / 2, abs(maxY - minY) / 2)
            # (maxX - minX) / 2, (maxY - minY) / 2
            # c1 = (abs(rect1[1][0] - rect1[0][0]) / 2 , abs(rect1[0][1] - rect1[1][1]) / 2)
            # c2 = (abs(rect1[1][0] - rect1[0][0]) / 2 , abs(rect1[0][1] - rect1[1][1]) / 2)
            d = Geometry.dist(c1, c2)
            dmax = max(d1, d2) / 2
            #dmax = dmax / 2
            #if (d < dmax):
            #    if (dmax > 50):
            #        union.append(rect1 if dmax == d1 else rect2)
            if (closeRects(rect1, rect2)):
                r = unionRects(rect1, rect2)
                if (largeEnough(r)):
                    union.append(r)
    if (len(union) == 0):
        return rects1
    return union

def unionClustToRect(clusters1, clusters2):
    rects1 = []
    rects2 = []
    for cluster in clusters1:
        if (len(cluster) < MagicConstants.minPinClust):
            continue
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects1.append([(minX, maxY), (maxX, minY)])
    for cluster in clusters2:
        if (len(cluster) < MagicConstants.minPinClust):
            continue
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects2.append([(minX, maxY), (maxX, minY)])
    rects = unionCloseRects(rects1, rects2)
    return rects

import glob

images = [file for file in glob.glob(route)]

rects = []
storage = []

for i in range(1, len(images)):
    # Iterate over first two frames
    img1 = images[i-1]
    img2 = images[i]
    if (first_two_frames): # Second two frames are tracked using detected info
        rects = tracker.trackRects(rects, img1)
        drawer = Drawer(img1, img2, [255, 0, 0])
        img = drawer.drawRects(rects, img2)
        if(show):
            plt.imshow(img, extent=[0, 300, 0, 1], aspect=200)
            plt.draw()
            plt.pause(0.001)
        if (len(rects) != 0):
            storage = rects
        else:
            rects = storage
        first_two_frames = False
        continue
    # Point selection
    selector = PointSelector(img1, img2)
    drawer = Drawer(img1, img2)

    good_new, good_old, road_new, road_old = selector.select()

    pts1 = good_old
    pts2 = good_new

    drawer.drawFlow(pts1, pts2, 'goodEnough.jpg', img2)
    outlierer = Outlierer()
    visOdom = VisOdometry(cam=cam, ptsold=pts1, ptsnew=pts2)

    # magnitude outliers
    out1, out2 = outlierer.optFLowMagnOutliers(pts1, pts2)
    clst1 = Cluster(out1, out2) # clusterpoints, clustervects

    #drawer.drawPoints(out1, 'out1.jpg', selector.optFlow.img1, True)

    # good outliers by essential matrix
    clst2 = Cluster(visOdom.outold, visOdom.outnew)

    #drawer.drawPoints(visOdom.outold, 'out2.jpg', selector.optFlow.img1, True)

    # epilines outliers
    outliers1 = outlierer.selectOutliers(pts1, visOdom.epilinesold)
    #drawer.drawPoints(outliers1, 'out3.jpg', selector.optFlow.img1, True)
    outliers2 = outlierer.selectOutliers(pts2, visOdom.epilinesnew)

    clst3 = Cluster(outliers1, outliers2)

    rects = unionClustToRect(clst1.clstnewpts, clst1.clstnewpts)
    rects1 = unionClustToRect(clst2.clstnewpts, clst2.clstnewpts)
    #bad outliers
    rects2 = unionClustToRect(clst3.clstnewpts, clst3.clstnewpts)
    rects = unionCloseRects(rects, rects1)

    reconst = Reconstructor(cam, visOdom.R, visOdom.t)
    p1 = [124, 258]
    p2 = [115, 254]
    p3 = [153, 254]
    p4 = [146, 250]
    points = np.array([p1,p2,p3,p4])
    #drawer.drawPoints(points,'lel.jpg', selector.optFlow.img1, True)
    p3d1 = reconst.reconstructPoint(p1,p2)
    p3d2 = reconst.reconstructPoint(p3,p4)
    lameDist = geom.dist3D(p3d1, p3d2)
    realDist = 70 # centimeter
    try:
        scaleFactor = realDist / lameDist
    except Exception:
        scaleFactor = 20
    #scaleFactor = 20
    print("scale"+str(scaleFactor))

    if (reconstruct):
        road1, road2 = selector.getPtsOnRoad()
        cv2.imwrite("bottom.jpg", selector.bot1)
        #drawer.drawFlow(road1, road2, 'out1out.jpg', selector.optFlow.img1, True)
        road_plane, pts = reconst.pointCloud(road1, road2, plot)
        #drawer.plotCloud(pts)
        #print(pts)
        road_pts1, road_pts2 = reconst.pixelOnRoad(selector.optFlow.img1, selector.optFlow.img2, road_plane)
        drawer.drawPoints(road_pts1, 'out.png', selector.optFlow.img1, True)
    #print("everything")
    #eq, everything = reconst.pointCloud(pts1, pts2, False)
    #drawer.plotCloud(everything)
    if (len(rects) != 0):
        storage = rects
    else:
        rects = storage

    img = drawer.drawRects(rects, img2)
    if (show):
        d = clst2.minDistToClusters(clst2.clstoldpts, clst2.clstnewpts, reconst, scaleFactor=scaleFactor)
        print(d)
        plt.imshow(img, extent=[0,600,0,2], aspect=200)
        plt.draw()
        plt.pause(0.02)
        cv2.imwrite('clusters.jpg', img)
        first_two_frames = True
    else:
        cv2.imwrite('clusters.jpg', img)
    print("lel")
    first_two_frames = True
    #break