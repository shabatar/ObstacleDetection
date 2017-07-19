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

clst = Cluster()
tracker = Tracker()
cnst = MagicConstants()
route = "data/*.png"

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
'''
# data5
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

cam = Camera(calibMat, distCoeffs, focal, pp, size)

def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

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
            d = dist(c1, c2)
            dmax = max(d1, d2) / 2
            #dmax = dmax / 2
            if (d < dmax):
                if (dmax > 50):
                    union.append(rect1 if dmax == d1 else rect2)
    if (len(union) == 0):
        return rects1
    return union

def unionCloseRects1(clusters1, clusters2):
    rects1 = []
    rects2 = []
    for cluster in clusters1:
        if (len(cluster) < cnst.minPinClust):
            continue
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects1.append([(minX, maxY), (maxX, minY)])
    for cluster in clusters2:
        if (len(cluster) < cnst.minPinClust):
            continue
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects2.append([(minX, maxY), (maxX, minY)])
    rects = unionCloseRects(rects1, rects2)
    return rects

def moduleVect(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def unitVect(vector):
    return vector / np.linalg.norm(vector)

def angleVect(v1, v2):
    v1_u = unitVect(v1)
    v2_u = unitVect(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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
                alpha = angleVect(np.array(v1), np.array(v2))
                if(alpha < cnst.critAlpha):
                    cnt += 1
        if(cnt <= len(cluster)**2//4): # bad cluster
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



def rectangles(clusters):
    rects = []
    for cluster in clusters:
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects.append([(minX, maxY), (maxX, minY)])
        #rects.append(cv2.rectangle(frame1, (minX, maxY), (maxX, minY), color, 3))
    return rects

def line_intersection(line1, line2):
    return 0

def dist3D(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

import glob

images = [file for file in glob.glob(route)]

first_two_frames = False

rects = []

for i in range(1, len(images)):
    img1 = images[i]
    img2 = images[i-1]
    if (first_two_frames):
        rects = tracker.trackRects(rects, img1)
        drawer = Drawer(img1, img2, [255, 0, 0])
        img = drawer.drawRects(rects, img2)
        plt.imshow(img, extent=[0, 300, 0, 1], aspect=200)

        plt.draw()
        plt.pause(0.01)
        first_two_frames = False
        continue
    selector = PointSelector(img1, img2, size)
    drawer = Drawer(img1, img2, [0, 0, 255])

    good_new, good_old, road_new, road_old = selector.select()

    pts1 = np.int32(good_old)
    pts2 = np.int32(good_new)

    road_new = np.int32(road_new)
    road_old = np.int32(road_old)

    drawer.drawFlow(pts1, pts2, 'goodEnough.jpg', img1)

    outlierer = Outlierer()


    # visual odometry
    # find essential matrix between frames
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=cam.focal, pp=cam.pp, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    # optical flow outliers
    out1, out2 = outlierer.optFLowMagnOutliers(pts1, pts2)
    clusteredout1 = clst.clusterPoints(out1, cnst.clusterConstant)
    clusteredout2 = clst.clusterPoints(out2, cnst.clusterConstant)

    frame1 = drawer.drawClusters(clusteredout1, img1, [0,0,255])
    frame2 = drawer.drawClusters(clusteredout2, img2, [0,0,255])

    #cv2.imwrite('optflowout1.jpg', frame1)
    #cv2.imwrite('optflowout2.jpg', frame2)

    # these are outliers
    if (mask is None):
        continue
    out1 = pts1[mask.ravel() == 0]
    out2 = pts2[mask.ravel() == 0]

    # cluster and draw them
    # good outliers
    clusteredout11 = clst.clusterPoints(out1, cnst.clusterConstant)
    clusteredout22 = clst.clusterPoints(out2, cnst.clusterConstant)

    frame1 = drawer.drawClusters(clusteredout11, img1, [0,0,255])
    frame2 = drawer.drawClusters(clusteredout22, img2, [0,0,255])

    #cv2.imwrite('clusters.jpg', frame1)
    #cv2.imwrite('clusters1.jpg', frame2)

    # to select inliers only
    pts11 = pts1[mask.ravel() == 1]
    pts22 = pts2[mask.ravel() == 1]

    # recover pose of camera, focal = fx
    _, R, t, mask3 = cv2.recoverPose(E, pts11, pts22, focal=cam.focal, pp=cam.pp)

    # outliers of this method
    out122 = pts11[mask3.ravel() == 0]
    out222 = pts22[mask3.ravel() == 0]


    # got R and t of camera now
    rotTrans = np.column_stack((R, t))
    projMat = np.dot(calibMat, rotTrans)

    # to find fundamental matrix and draw epilines
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    if (F is None):
        continue
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)


    # epilines outliers
    outliers1 = outlierer.selectOutliers(pts1, lines2)
    outliers2 = outlierer.selectOutliers(pts2, lines1)

    #ЗАХАРДКОДИМ?
    clusteredout0 = clst.clusterPoints(outliers1, cnst.clusterConstant)
    clusteredout = clst.clusterPoints(outliers2, cnst.clusterConstant)

    frame1 = drawer.drawClusters(clusteredout0, img1, [0,0,255])
    frame2 = drawer.drawClusters(clusteredout, img2, [0,0,255])

    #cv2.imwrite('sup.jpg', frame1)
    #cv2.imwrite('sup2.jpg', frame2)

    #clusteredout (for sup), clusteredout11 (for optflowmagn) and clusteredout1 (for clusters.jpg)
    clv = clusterVect(clustersnew=clusteredout, clustersold=clusteredout0)
    clv = removeOutliers(clv)
    newclv, oldclv = recoverClusters(clusteredout, clv)
    clv1 = clusterVect(clustersold=clusteredout11, clustersnew=clusteredout22)
    clv1 = removeOutliers(clv1)
    newclv1, oldclv1 = recoverClusters(clusteredout22, clv1)
    clv2 = clusterVect(clustersnew=clusteredout2, clustersold=clusteredout1)
    clv2 = removeOutliers(clv2)
    newclv2, oldclv2 = recoverClusters(clusteredout2, clv2)
    #out = outlierer.
    #rects = unionCloseRects1(clusteredout22, clusteredout22)
    #rects1 = unionCloseRects1(clusteredout, clusteredout11)
    #rects = unionCloseRects(rects, rects1)
    #rects = unionCloseRects1(clv, clv1)
    rects = unionCloseRects1(newclv1,newclv1)
    rects1 = unionCloseRects1(newclv2, newclv2)
    rects = unionCloseRects(rects, rects1)
    #rects1 = unionCloseRects1(clusteredout, clusteredout2)
    #rects = unionCloseRects(rects, rects1)
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

    #plt.imshow(img, extent=[0,600,0,2], aspect=200)

    reconst = Reconstructor(cam, R, t)

    road_plane = reconst.pointCloud(road_old, road_new)
    #print(selector.bot_img1)
    #cv2.imwrite("bottom.jpg", selector.bot_img2)
    #print(road_old)
    road_pts1, road_pts2 = reconst.pixelOnRoad(selector.img1, selector.img2, road_plane)
    road_pts1 = np.int32(road_pts1)
    road_pts2 = np.int32(road_pts2)
    #print(road_pts1)
    drawer.drawPoints(road_pts1, 'out.jpg', selector.img1, True)
    print("lel")
    #selector.selectRoad()
    #plt.draw()
    #plt.pause(0.001)
    #break

    #first_two_frames = True
    #plt.show()
    #cv2.imwrite(os.path.join('detected',img1),img)