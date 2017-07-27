'''
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)
size = img1.shape
newh = size[0] // 3
img1 = img2[newh: size[0], 1: size[1]]
img2 = img2[newh: size[0], 1: size[1]]

cv2.imwrite('0000000000.png', img1)
cv2.imwrite('0000000001.png', img2)

img1, img2 = '0000000000.png', '0000000001.png'
'''

#good_new, good_old = lucasKanade(img1, img2)

# choose from each regions good random points
#good_new, good_old, road_new, road_old = goodEnoughPoints(img1, img2, good_new, good_old)

#U,s,V = linalg.svd(F)


#img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)


#allColors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100], [50, 0, 100], [50, 100, 0], [0, 50, 100], [0, 100, 50], [100, 0, 50], [100, 50, 0], [200, 50, 0], [200, 0, 50], [0, 200, 50], [0, 50, 200]]

#img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

#cv2.imwrite('frame2.jpg',img3)
#cv2.imwrite('frame3.jpg',img5)

# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)


# draw  outliers
#drawFlow(out1, out2, 'outliers1.jpg', img2)

# draw another outliers
#drawFlow(out1, out2, 'outliers2.jpg', img2)




# below comments only
# wrong way
'''
R1, R2, P1, P2, Q, valid1, valid2 = cv2.stereoRectify(calibMat, distCoeffs, calibMat, distCoeffs, size, R, t)
# P1 and P2 are projection matrices of camera frames

pts1 = good_old
pts2 = good_new
# [[x y], ...] -> [[x,...], [y,...]]
pts1x = []
pts1y = []
for e1 in pts1:
    pts1x.append(e1[0])
    pts1y.append(e1[1])
pts2x = []
pts2y = []
for e2 in pts2:
    pts2x.append(e2[0])
    pts2y.append(e2[1])
pts1 = np.array([pts1x, pts1y])
pts2 = np.array([pts2x, pts2y])

Points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)


# then plot them
Points4Dx = Points4D[0]
Points4Dy = Points4D[1]
Points4Dz = Points4D[2]
Points4Dw = Points4D[3]
Points3D = []
for i in range(0,Points4Dx.shape[0]):
    Points4Dx[i] = Points4Dx[i] / Points4Dw[i]
    Points4Dy[i] = Points4Dy[i] / Points4Dw[i]
    Points4Dz[i] = Points4Dz[i] / Points4Dw[i]
    Point3D = [Points4Dx[i], Points4Dy[i], Points4Dz[i]]
    Points3D.append(Point3D)
#Points3Ds = [Points4Dx,Points4Dy,Points4Dz]]
roadPLane = random.sample(Points3D, 3)
#print(Points3D)
#print(roadPLane)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Points4Dx, Points4Dy, Points4Dz, c='r', marker='*')
plt.savefig('fig.png')
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.show()
'''

# for p1, p2 in flow:
# point3D = self.reconstructPoint(p1, p2)
# A, B, C, D = plane[0], plane[1], plane[2], plane[3]
# x, y, z = point3D[0], point3D[1], point3D[2]
# if (np.any(A * x + B * y + C * z + D) < 3):
#    print("kek")
#    road_pts1.append(p1)
#    road_pts2.append(p2)
'''
point1 = [point1[0], point1[1]]
point2 = [point2[0], point2[1]]
point3D = self.reconstructPoint(point1, point2)
A, B, C, D = plane[0], plane[1], plane[2], plane[3]
x, y, z = point3D[0], point3D[1], point3D[2]
if(np.any(A*x + B*y + C*z + D) < 3):
    print("kek")
    road_pts1.append(point1)
    road_pts2.append(point2)
'''
# return road_pts1, road_pts2




'''
    def __init__(self, img1, img2, size, read=False):
        if (not read):
            self.img1 = cv2.imread(img1)
            self.img1 = cv2.blur(self.img1, cnst.gaussWin)
        else:
            self.img1 = img1
        #self.img1 = cv2.blur(self.img1, cnst.gaussWin)
        #self.img1 = cv2.threshold(self.img1,127,255,cv2.THRESH_BINARY)
        #self.img1 = cv2.pyrMeanShiftFiltering(self.img1, cnst.meanShiftN, cnst.meanShiftN)
        if (not read):
            self.img2 = cv2.imread(img2)
            self.img2 = cv2.blur(self.img2, cnst.gaussWin)
        else:
            self.img2 = img2
        #self.img2 = cv2.blur(self.img2, cnst.gaussWin)
        #self.img2 = cv2.threshold(self.img2,127,255,cv2.THRESH_BINARY)
        #self.img2 = cv2.pyrMeanShiftFiltering(self.img2, cnst.meanShiftN, cnst.meanShiftN)
        self.magicConstant = cnst.pointToSelect
        self.imgSize = size


            p1 = pts1[i]  # pts1 = points on frame1
            p1 = np.array([p1[0] + w / 2, p1[1] + h / 2, z])  # p1 in camera1 basis
            x1, y1, z1 = p1[0], p1[1], p1[2]

            p2 = pts2[i]  # pts2 = points on frame 2
            p2 = np.array([p2[0] + w / 2, p2[1] + h / 2, z])  # p2 in camera2 basis
            p2 = np.dot(self.rotMat, p2)  # p2 in camera1 basis

            x2, y2, z2 = p2[0], p2[1], p2[2]

            tx, ty, tz = self.trVect[0], self.trVect[1], self.trVect[2]  # translation vector

            S1 = (ty * x1 - tx * y1) / (x2 * y1 - y2 * x1)
            S2 = (tz * x1 - tx * z1) / (x2 * z1 - z2 * x1)
            T1 = (tx + S1 * x2) / x1
            T2 = (tx + S2 * x2) / x1
            x0, y0, z0 = [0], [0], [0]
            #print(abs((T1 * x1) - (tx + S1 * x2))[0])
            if (np.all(abs((T1 * x1) - (tx + S1 * x2))) < EPS):
                x0 = T1 * x1
            if (np.all(abs((T1 * y1) - (ty + S1 * y2))) < EPS):
                y0 = T1 * x1
            if (np.all(abs((T1 * z1) - (tz + S1 * z2))) < EPS):
                z0 = T1 * x1
            if (abs((T2 * x1) - (tx + S2 * x2)) < EPS):
                x0 = T2 * x1
            if (abs((T2 * y1) - (ty + S2 * y2)) < EPS):
                y0 = T2 * x1
            if (np.all(abs((T2 * z1) - (tz + S2 * z2))) < EPS):
                z0 = T2 * x1
            points3Dx.append(x0)
            points3Dy.append(y0)
            points3Dz.append(z0)
            pts3Ds.append([x0[0], y0[0], z0[0]])

    def cornerHarris(self):
        img = self.img2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # 0.04
        dst = cv2.cornerHarris(gray, 4, 3, 0.09)
        dst = cv2.dilate(dst, None)
        ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
        dst = np.uint8(dst)

        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        res = np.hstack((centroids, corners))
        res = np.int0(res)
        try:
            img[res[:, 1], res[:, 0]] = [0, 0, 255]
            img[res[:, 3], res[:, 2]] = [0, 255, 0]
        except Exception:
            print("kek")
            corners = corners[:, np.newaxis]
            return corners

        # cv2.imwrite('harris.png',img)
        corners = corners[:, np.newaxis]
        return corners

ANOTHER
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


    def lucasKanade(self):
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        img_yuv = cv2.cvtColor(self.img1, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.img1 = img_output

        img_yuv = cv2.cvtColor(self.img2, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.img2 = img_output


        old_frame = self.img1
        frame = self.img2
        size = self.img1.shape
        newh = size[0] // 3
        old_frame = old_frame[newh: size[0], 1: size[1]]
        frame = frame[newh: size[0], 1: size[1]]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        old_frame = self.img1
        frame = self.img2
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('kek.png', old_frame)
        #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        p0 = self.cornerHarris()

        mask = np.zeros_like(old_frame)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), [0, 0, 255], 2)
            frame = cv2.circle(frame, (a, b), 2, [0, 0, 255], -1)
        img = cv2.add(frame, mask)
        # cv2.imwrite('opticalFlow.jpg', img)
        return good_new, good_old
    '''













''' two sets intersection
pts1set = set([tuple(x) for x in pts1])
pts2set = set([tuple(x) for x in pts2])
roadpts1set = set([tuple(x) for x in roadpts1])
roadpts2set = set([tuple(x) for x in roadpts2])

roadpts1 = np.array([x for x in pts1set & roadpts1set])
roadpts2 = np.array([x for x in pts2set & roadpts2set])
'''
'''  кусок кода продублированный выше
# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
#print(lines1)
lines1 = lines1.reshape(-1,3)
print(lines1)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

cv2.imwrite('frame2.jpg',img3)
cv2.imwrite('frame3.jpg',img5)
'''
''' плохая рекурсия
    if (len(nearby) == 1):
        points.remove(point)
        return []

    elif len(points) < 10:
        return nearby

    else:
        map(lambda p : points.remove(p), nearby)
        for p in nearby:
            nearby = nearby + getNearbyPoints(points, p, eps)
    return nearby
'''
