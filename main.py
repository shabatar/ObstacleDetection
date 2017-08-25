import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
from pointselector import PointSelector
from drawer import Drawer
from outlierer import Outlierer
from camera import Camera
from constants import MagicConstants
from tracker import Tracker
from reconstructor import Reconstructor
from cluster import Cluster
from visodometry import VisOdometry
from visodometry import Geometry
from rectangle import *

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

tracker = Tracker()
route = "data/00000000*.png"
first_two_frames = False

# If need to show detected
SHOW = True
# If need to use road reconstruction
RECONSTRUCT = False
# If need to use reconstructed plotting
PLOT = False
# If know scalefactor
SCALE = False
# If need to track
TRACK = False

cam = Camera(calibMat, distCoeffs, focal, pp, size)
images = [file for file in glob.glob(route)]
rects = []
storage = []
scaleFactor = 0

def calcScaleFactor(flag, scaleFactor, reconst):
    if (flag):
        return scaleFactor
    p1 = [124, 258]
    p2 = [115, 254]
    p3 = [153, 254]
    p4 = [146, 250]
    p3d1 = reconst.reconstructPoint(p1, p2)
    p3d2 = reconst.reconstructPoint(p3, p4)
    lameDist = Geometry.dist3D(p3d1, p3d2)
    realDist = 70  # centimeter
    if (lameDist < MagicConstants.reconstructEPS):
        scaleFactor = 20
    else:
        scaleFactor = realDist / lameDist
    #print("scale="+str(scaleFactor))
    return scaleFactor

for i in range(1, len(images)):
    # Iterate over first two frames
    img1 = images[i-1]
    img2 = images[i]

    if (first_two_frames): # Second two frames are tracked using detected info
        rects = tracker.trackRects(rects, img1)
        drawer = Drawer(img1, img2)
        img = drawer.drawRects(rects, img2)
        if (SHOW):
            plt.imshow(img, extent=[0, 300, 0, 1], aspect=200)
            plt.draw()
            plt.pause(0.001)
        if (len(rects) != 0):
            storage = rects
        else:
            rects = storage
        first_two_frames = False
    # Point selection
    selector = PointSelector(img1, img2)
    drawer = Drawer(img1, img2)

    pts1, pts2 = selector.select() # new, old

    drawer.drawFlow(pts1, pts2, 'goodEnough.jpg', img2)
    outlierer = Outlierer()
    visOdom = VisOdometry(cam=cam, ptsold=pts2, ptsnew=pts1)

    # magnitude pts
    out1, out2 = outlierer.optFLowMagnOutliers(pts1, pts2) # new, old
    clst1 = Cluster(out2, out1)

    # points by essential matrix
    clst2 = Cluster(visOdom.outold, visOdom.outnew)
    rects = unionRects(clst1.unionClustToRect(clst1.clstnewpts, clst1.clstnewpts),
                       clst1.unionClustToRect(clst2.clstnewpts, clst2.clstnewpts))

    reconst = Reconstructor(cam, visOdom.R, visOdom.t)

    scaleFactor = calcScaleFactor(SCALE, scaleFactor, reconst)
    SCALE = True

    if (RECONSTRUCT):
        road1, road2 = selector.getPtsOnRoad()
        road_plane, pts = reconst.pointCloud(road1, road2, PLOT)
        road_pts1, road_pts2 = reconst.pointsOnRoad(selector.optFlow.img1, selector.optFlow.img2, road_plane)
        #...

    if (len(rects) != 0):
        storage = rects
    else:
        rects = storage

    img = drawer.drawRects(rects, img2)
    if (SHOW):
        d = clst2.minDistToClusters(clst2.clstoldpts, clst2.clstnewpts, reconst, scaleFactor=scaleFactor)
        plt.imshow(img, extent=[0, 600, 0, 2], aspect=200)
        plt.draw()
        plt.pause(0.02)
        if (TRACK):
            first_two_frames = True
    else:
        cv2.imwrite('clusters.jpg', img)
    if (TRACK):
        first_two_frames = True
    #break