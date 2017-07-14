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

cnst = MagicConstants()
route = "data/0000000*.png"

def dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def getNearbyPoints(unassigned, point, eps):
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
        #map(lambda p: unassigned.remove(p), nearby)

    return nearby

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
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects1.append([(minX, maxY), (maxX, minY)])
    for cluster in clusters2:
        maxX = max(cluster, key=lambda p: p[0])[0]
        maxY = max(cluster, key=lambda p: p[1])[1]
        minX = min(cluster, key=lambda p: p[0])[0]
        minY = min(cluster, key=lambda p: p[1])[1]
        rects2.append([(minX, maxY), (maxX, minY)])
    rects = unionCloseRects(rects1, rects2)
    return rects


def clusterPoints(points, eps):  # eps = num of pixels to be considered as close enough
    unassigned = [(point[0], point[1]) for point in points]
    clusters = []
    while (len(unassigned)):
        for p in unassigned:
            myPoint = (p[0], p[1])
            # eps=52 is a good number
            near = getNearbyPoints(unassigned, myPoint, eps)
            clusters.append(near)
    return clusters

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

def trackRect(rect, frame):
    # take first frame of the video
    frame = cv2.imread(frame)
    # setup initial location of window
    minX, maxY, maxX, minY = rect[0][0], rect[0][1], rect[1][0], rect[1][1]
    c = minX
    r = minY
    w = maxX - minX
    h = maxY - minY
    track_window = (c, r, w, h)
    if (c < 0 or r < 0 or w < 0 or h < 0):
        return [(1, 2), (3, 4)]
    if (minX == maxX or minY == maxY):
        return [(1, 2), (3, 4)]
    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    x, y, w, h = track_window
    #(minX, maxY), (maxX, minY)
    img2 = cv2.rectangle(frame, (x, y+h), (x + w, y), 255, 2)
    #cv2.imwrite('img2.png', img2)
    rect = [(x, y+h), (x + w, y)]
    return rect

def trackRects(rectas, img):
    rects = []
    for rect in rectas:
        rect1 = trackRect(rect,img)
        rects.append(rect1)
    return rects


#x = xa1 + (xa2 - xa1) * ta = xb1 + (xb2 - xb1) * tb
#and
#y = ya1 + (ya2 - ya1) * ta = yb1 + (yb2 - yb1) * tb


def line_intersection(line1, line2):
    return 0

def dist3D(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

def pointCloud(cam, rotMat, trVect, pts1, pts2):
    EPS = 0.001
    points3Dx = []
    points3Dy = []
    points3Dz = []
    z = cam.focal
    w,h = cam.imgsize
    print(trVect)
    for i in range(0, len(pts1)):
        p1 = pts1[i] # pts1 = points on frame1
        p1 = np.array([p1[0] + w/2, p1[1] + h/2, z]) # p1 in camera1 basis
        x1, y1, z1 = p1[0], p1[1], p1[2]

        p2 = pts2[i] # pts2 = points on frame 2
        p2 = np.array([p2[0] + w/2, p2[1] + h/2, z]) # p2 in camera2 basis
        p2 = np.dot(rotMat, p2) # p2 in camera1 basis

        x2, y2, z2 = p2[0], p2[1], p2[2]

        tx, ty, tz = trVect[0], trVect[1], trVect[2] # translation vector

        S1 = (ty * x1 - tx * y1) / (x2 * y1 - y2 * x1)
        S2 = (tz * x1 - tx * z1) / (x2 * z1 - z2 * x1)
        T1 = (tx + S1 * x2) / x1
        T2 = (tx + S2 * x2) / x1

        if (abs((T1 * x1) - (tx + S1 * x2)) < EPS):
            print("t1x")
            x0 = T1 * x1
        if (abs((T1 * y1) - (ty + S1 * y2)) < EPS):
            print("t1y")
            y0 = T1 * x1
        if (abs((T1 * z1) - (tz + S1 * z2)) < EPS):
            print("t1z")
            z0 = T1 * x1
        #if (abs((T2 * x1) - (tx + S2 * x2)) < EPS):
            #print("t2x")
            #x0 = T2 * x1
        #if (abs((T2 * y1) - (ty + S2 * y2)) < EPS):
            #print("t2y")
            #y0 = T2 * x1
        if (abs((T2 * z1) - (tz + S2 * z2)) < EPS):
            print("t2z")
            z0 = T2 * x1
        '''
        if (abs((T1 * x1) - (tx + S1 * x2)) < EPS and
                    abs((T1 * y1) - (ty + S1 * y2)) < EPS and
                    abs((T1 * z1) - (tz + S1 * z2)) < EPS):
            x0 = T1 * x1
            y0 = T1 * y1
            z0 = T1 * z1
        elif (abs((T2 * x1) - (tx + S2 * x2)) < EPS and
                      abs((T2 * y1) - (ty + S2 * y2)) < EPS and
                      abs((T2 * z1) - (tz + S2 * z2)) < EPS):
            x0 = T2 * x1
            y0 = T2 * y1
            z0 = T2 * z1
        else:
            print("не пересеклись :(")
            return 0
        '''
        print([x0, y0, z0])
        points3Dx.append(x0)
        points3Dy.append(y0)
        points3Dz.append(z0)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points3Dx, points3Dy, points3Dz, c='r', marker='*')
    plt.savefig('fig.png')
    xmin = min(points3Dx)
    ymin = min(points3Dy)
    xmax = max(points3Dx)
    ymax = max(points3Dy)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    return 1

import glob

images = [file for file in glob.glob(route)]

first_two_frames = False

rects = []

for i in range(1, len(images)):
    img1 = images[i]
    img2 = images[i-1]
    if (first_two_frames):
        rects = trackRects(rects, img1)
        drawer = Drawer(img1, img2, [255, 0, 0])
        img = drawer.drawRects(rects, img2)
        plt.imshow(img, extent=[0, 300, 0, 1], aspect=200)

        plt.draw()
        plt.pause(0.01)
        continue
    selector = PointSelector(img1, img2)
    drawer = Drawer(img1, img2, [0, 0, 255])

    good_new, good_old, road_new, road_old = selector.select()

    pts1 = np.int32(good_old)
    pts2 = np.int32(good_new)

    road_new = np.int32(road_new)
    road_old = np.int32(road_old)

    drawer.drawFlow(pts1, pts2, 'goodEnough.jpg', img1)
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

    cam = Camera(calibMat, distCoeffs, focal, pp, size)
    outlierer = Outlierer()


    # visual odometry
    # find essential matrix between frames
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=cam.focal, pp=cam.pp, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    # optical flow outliers
    out1, out2 = outlierer.optFLowMagnOutliers(pts1, pts2)
    clusteredout1 = clusterPoints(out1, cnst.clusterConstant)
    clusteredout2 = clusterPoints(out2, cnst.clusterConstant)

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
    clusteredout11 = clusterPoints(out1, cnst.clusterConstant)
    clusteredout22 = clusterPoints(out2, cnst.clusterConstant)

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

    points3D = pointCloud(cam, R, t, out1, out2)



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
    clusteredout0 = clusterPoints(outliers1, cnst.clusterConstant)
    clusteredout = clusterPoints(outliers2, cnst.clusterConstant)

    frame1 = drawer.drawClusters(clusteredout0, img1, [0,0,255])
    frame2 = drawer.drawClusters(clusteredout, img2, [0,0,255])

    #cv2.imwrite('sup.jpg', frame1)
    #cv2.imwrite('sup2.jpg', frame2)

    #clusteredout (for sup), clusteredout11 (for optflow) and clusteredout1 (for clusters.jpg)

    #rects = unionCloseRects1(clusteredout11, clusteredout1)
    #rects1 = unionCloseRects1(clusteredout0, clusteredout11)
    #rects = unionCloseRects(rects, rects1)
    #img = drawer.drawRects(rects, img1)
    #cv2.imwrite(os.path.join('detected',img1),img)
    #plt.imshow(img)
    #plt.show()

    #out = outlierer.
    rects = unionCloseRects1(clusteredout22, clusteredout22)
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

    #plt.draw()
    #plt.pause(0.01)

    #first_two_frames = True
    #plt.show()
    #cv2.imwrite(os.path.join('detected',img1),img)