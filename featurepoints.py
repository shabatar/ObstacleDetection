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

clst = Cluster()
tracker = Tracker()
route = "data/*.png"
geom = Geometry()

# set parameters for camera (set of data)
calibMat = np.array([
    [9.895267e+02, 0.000000e+00, 7.020000e+02],
    [0.000000e+00, 9.878386e+02, 2.455590e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
size = (1392, 512)
distCoeffs = (-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02)
focal = 9.895267e+02
pp = (7.020000e+02, 2.455590e+02)

# data1
'''
corner_dist = 9.950000e-02
size = (1.392000e+03,5.120000e+02)
calibMat = np.array([
    [9.842439e+02, 0.000000e+00, 6.900000e+02],
    [0.000000e+00, 9.808141e+02, 2.331966e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
distCoeffs = (-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02)
focal = 9.842439e+02
pp = (6.900000e+02,2.331966e+02)
'''
# data2
'''
size = (1.392000e+03, 5.120000e+02)
calibMat = np.array([
    [9.799200e+02, 0.000000e+00, 6.900000e+02],
    [0.000000e+00, 9.741183e+02, 2.486443e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
distCoeffs = (-3.745594e-01, 2.049385e-01, 1.110145e-03, 1.379375e-03, -7.084798e-02)
focal = 9.799200e+02
pp = (6.900000e+02, 2.486443e+02)

# data3
size = (1.392000e+03, 5.120000e+02)
calibMat = np.array([
    [9.842439e+02, 0.000000e+00, 6.900000e+02],
    [0.000000e+00, 9.808141e+02, 2.331966e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
distCoeffs = (-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02)
focal = 9.842439e+02
pp = (6.900000e+02,2.331966e+02)
'''

# data5
'''
size = (2048, 1024)
calibMat = np.array([
    [2268.36, 0.000000e+00, 1048.64],
    [0.000000e+00, 2312.0, 519.27],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
distCoeffs = (-3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02)
focal = 2268.36
pp = (1048.64,519.27)
'''
# data6 and 7 and 8 calib time 28 09
'''
calibMat = np.array([
    [9.812178e+02, 0.000000e+00, 6.900000e+02],
    [0.000000e+00, 9.758994e+02, 2.471364e+02],
    [0.000000e+00, 0.000000e+00, 1.000000e+00]
])
size = (1392, 512)
distCoeffs = (-3.791375e-01, 2.148119e-01, 1.227094e-03, 2.343833e-03, -7.910379e-02)
focal = 9.812178e+02
pp = (6.900000e+02, 2.471364e+02)
'''
cam = Camera(calibMat, distCoeffs, focal, pp, size)

def unionCloseRects(rects1, rects2):
    union = []
    for rect1 in rects1:
        for rect2 in rects2:
            minX, maxY, maxX, minY = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1]
            d1 = math.sqrt((maxX-minX) ** 2 + (maxY-minY) ** 2)
            c1 = (abs(maxX - minX) / 2, abs(maxY - minY) / 2)
            minX, maxY, maxX, minY = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1]
            d2 = math.sqrt((maxX-minX) ** 2 + (maxY-minY) ** 2)
            c2 = (abs(maxX - minX) / 2, abs(maxY - minY) / 2)
            # (maxX - minX) / 2, (maxY - minY) / 2
            # c1 = (abs(rect1[1][0] - rect1[0][0]) / 2 , abs(rect1[0][1] - rect1[1][1]) / 2)
            # c2 = (abs(rect1[1][0] - rect1[0][0]) / 2 , abs(rect1[0][1] - rect1[1][1]) / 2)
            d = geom.dist(c1, c2)
            dmax = max(d1, d2) / 2
            #dmax = dmax / 2
            if (d < dmax):
                if (dmax > 50):
                    union.append(rect1 if dmax == d1 else rect2)
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

def clusterVect(clustersold, clustersnew):
    clusters = []
    for clust1, clust2 in zip(clustersold, clustersnew):
        clust = []
        for p1, p2 in zip(clust1, clust2):
            v = [p1[0]-p2[0], p1[1]-p2[1]]
            clust.append(v)
        clusters.append(clust)
    return clusters

def removeOutliers(clustervects):
    outclusters = []
    for cluster in clustervects:
        cnt = 0
        for v1 in cluster:
            for v2 in cluster:
                alpha = geom.angleVect(np.array(v1), np.array(v2))
                if(alpha < MagicConstants.critAlpha):
                    cnt += 1
        if(cnt <= (len(cluster)**2) // 4): # bad cluster
            outclusters.append([0])
            continue
        else:
            outclusters.append(cluster)
    return outclusters

def recoverClusters(newclust, vectclust): #recover clusters without outliers
    newclusters = []
    oldclusters = []
    for clust1, clust2 in zip(newclust, vectclust):
        if(len(clust2) == 1): #it's a point, bro! old clusters should not be recovered
            continue
        else:
            oldclust = []
            for p1, p2 in zip(clust1, clust2):
                newp = [p2[0]+p1[0], p2[1]+p1[1]]
                oldclust.append(newp)
            oldclusters.append(oldclust)
            newclusters.append(clust1)
    return newclusters, oldclusters

import glob

images = [file for file in glob.glob(route)]

first_two_frames = False

rects = []
storage = []
show = True

for i in range(1, len(images)):
    img1 = images[i]
    img2 = images[i-1]
    if (first_two_frames):
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
    selector = PointSelector(img1, img2)
    drawer = Drawer(img1, img2, [0, 0, 255])

    good_new, good_old, road_new, road_old = selector.select()

    pts1 = good_old
    pts2 = good_new
    drawer.drawFlow(pts1, pts2, 'goodEnough.jpg', img1)

    outlierer = Outlierer()
    visOdom = VisOdometry(cam=cam, ptsold=pts1, ptsnew=pts2)

    # shitty code starts here
    # magnitude outliers
    try:
        out1, out2 = outlierer.optFLowMagnOutliers(pts1, pts2)
        clusteredout1 = clst.clusterPoints(out1, MagicConstants.clusterConstant)
        clusteredout2 = clst.clusterPoints(out2, MagicConstants.clusterConstant)

        # good outliers by essential matrix
        clusteredout11 = clst.clusterPoints(visOdom.outold, MagicConstants.clusterConstant)
        clusteredout22 = clst.clusterPoints(visOdom.outnew, MagicConstants.clusterConstant)
    except Exception:
        continue

    '''
    # epilines outliers
    outliers1 = outlierer.selectOutliers(pts1, visOdom.epilinesnew)
    outliers2 = outlierer.selectOutliers(pts2, visOdom.epilinesold)

    #ЗАХАРДКОДИМ?
    clusteredout0 = clst.clusterPoints(outliers1, MagicConstants.clusterConstant)
    clusteredout = clst.clusterPoints(outliers2, MagicConstants.clusterConstant)

    #clusteredout (for epilines), clusteredout11 (for optflowmagn) and clusteredout1 (for essenmat)
    clv = clusterVect(clustersnew=clusteredout, clustersold=clusteredout0)
    clv = removeOutliers(clv)
    newclv, oldclv = recoverClusters(clusteredout, clv)
    '''
    clv1 = clusterVect(clustersold=clusteredout11, clustersnew=clusteredout22)
    clv1 = removeOutliers(clv1)
    newclv1, oldclv1 = recoverClusters(clusteredout22, clv1)
    clv2 = clusterVect(clustersnew=clusteredout2, clustersold=clusteredout1)
    clv2 = removeOutliers(clv2)
    newclv2, oldclv2 = recoverClusters(clusteredout2, clv2)

    #rects = unionCloseRects1(clusteredout22, clusteredout22)
    #rects1 = unionCloseRects1(clusteredout11, clusteredout11)
    #rects = unionCloseRects(rects, rects1)

    rects = unionClustToRect(newclv2, newclv2)
    rects1 = unionClustToRect(newclv1, newclv1)
    rects = unionCloseRects(rects, rects1)
    if (len(rects) != 0):
        storage = rects
    else:
        rects = storage

    img = drawer.drawRects(rects, img2)
    '''
    img = cv2.imread(img2, 0)
    img = cv2.blur(img, (2, 2))
    edges = cv2.Canny(img, 100, 200)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.draw()
    plt.pause(0.0125)
    '''
    if (show):
        plt.imshow(img, extent=[0,600,0,2], aspect=200)
        plt.draw()
        plt.pause(0.001)

    #reconst = Reconstructor(cam, visOdom.R, visOdom.t)
    #road1, road2 = selector.getPtsOnRoad()
    #drawer.drawFlow(road1, road2, 'out1out.jpg', selector.bot1, True)
    #road_plane = reconst.pointCloud(road1, road2)
    #cv2.imwrite("bottom.jpg", selector.bot1)
    #road_pts1, road_pts2 = reconst.pixelOnRoad(selector.optFlow.img1, selector.optFlow.img2, road_plane)
    #drawer.drawPoints(road_pts1, 'out.jpg', selector.optFlow.img1, True)

    #reconst.reconstructPoint1([3,3])
    print("lel")
    first_two_frames = True