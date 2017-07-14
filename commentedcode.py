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
