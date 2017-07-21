import cv2
import numpy as np
from constants import MagicConstants
from drawer import *
import random
import math

class Reconstructor:
    def __init__(self, cam, rotMat, trVect):
        self.cam = cam
        self.rotMat = rotMat
        self.trVect = trVect

    def planeByPoints(self, pt1, pt2, pt3):
        x1,y1,z1 = pt1[0], pt1[1], pt1[2]
        x2,y2,z2 = pt2[0], pt2[1], pt2[2]
        x3,y3,z3 = pt3[0], pt3[1], pt3[2]
        # Ax+By+Сz+D=0
        # det{ {x-x1, y-y1, z-z1}, {x2-x1, y2-y1, z2-z1}, {x3-x1, y3-y1, z3-z1} } = 0
        # x2-x1 = a2, y2-y1 = b2, z2-z1 = c2 etc.
        A = (y2-y1)*(z3-z1)-(y3-y1)*(z2-z1)
        B = (z2-z1)*(x3-x1)-(z3-z1)*(x2-x1)
        C = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
        D = x1*(-A)+y1*(-B)+z1*(-C)
        return A,B,C,D

    def distPlanePt(self, plane, pt):
        plane = np.array(plane)
        pt = np.array(pt)
        pt = pt.tolist()
        plane = plane.tolist()
        #if (type(pt[0]) is list):
        #    pt = [i[0] for i in plane]
        ptx, pty, ptz = pt[0], pt[1], pt[2]
        A, B, C, D = plane[0], plane[1], plane[2], plane[3]
        if (A == 0.0 and B == 0.0 and C == 0.0):
            #print("kek net ploskosti")
            return 10000
        up = abs(A*ptx + B*pty + C*ptz + D)
        down = math.sqrt(A**2 + B**2 + C**2)
        dst = up / down
        return dst

    def planeFitRansac(self, points, equation=True):
        best3pts = []
        bestPlane = 0, 0, 0, 0
        just3pts = random.sample(points, 3)
        plane = self.planeByPoints(just3pts[0], just3pts[1], just3pts[2])
        sumBest = sum(self.distPlanePt(plane, pt) for pt in points)
        bestPlane = plane
        for i in range(0, 50):
            just3pts = random.sample(points, 3)
            plane = self.planeByPoints(just3pts[0], just3pts[1], just3pts[2])
            sumofSqDist = sum([self.distPlanePt(plane, pt) for pt in points])
            if sumofSqDist < sumBest:
                sumBest = sumofSqDist
                best3pts = just3pts
                bestPlane = plane
        if(equation == False):
            point = random.sample(best3pts, 1)
            normal = bestPlane[0], bestPlane[1], bestPlane[2]
            return point, normal
        return bestPlane

    def PCA(self, data, correlation=False, sort=True):
        mean = np.mean(data, axis=0)
        data_adjust = data - mean
        # matrix = np.corrcoef(data_adjust.T)
        #: the data is transposed due to np.cov/corrcoef syntax
        if correlation:
            matrix = np.corrcoef(data_adjust.T)

        else:
            matrix = np.cov(data_adjust.T)

        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        if sort:
            #: sort eigenvalues and eigenvectors
            sort = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:, sort]

        return eigenvalues, eigenvectors

    def outlierRemove(self, kdtree, k=20, z_max=80, y_max=90, x_max=80):
        distances, i = kdtree.query(kdtree.data, k=k)
        from scipy.stats import zscore
        z_distances = zscore(np.mean(distances, axis=1))
        zfilter = abs(z_distances) < z_max
        return zfilter

    def bestFitPlane(self, points, equation=True):
        from scipy.spatial import KDTree
        points = [np.array(point) for point in points]
        points = np.array(points)
        points = KDTree(points, leafsize=points.shape[0]+1)
        filter = self.outlierRemove(points)
        points1 = []
        for i in range(0, points.data.shape[0]):
            point = points.data[i]
            if (filter[i] == True):
                points1.append([point[0], point[1], point[2]])
        points = points1
        w, v = self.PCA(points)
        import random
        points = random.sample(points, MagicConstants.roadPlanePts)
        #: the normal of the plane is the last eigenvector
        normal = v[:, 2]

        #: get a point from the plane
        point = np.mean(points, axis=0)

        if equation:
            a, b, c = normal
            d = -(np.dot(normal, point))
            for p in points:
                continue
                #print(a*p[0]+b*p[1]+c*p[2]+d)
            return a, b, c, d
        else:
            return point, normal

    def reconstructPoint1(self, point1, point2):
        z = 1
        w, h = (200,200)
        rotMat = np.array([
            [1.000000e+00, 0.000000e+00, 0.000000e+00],
            [0.000000e+00, math.sqrt(2) / 2, - math.sqrt(2) / 2],
            [0.000000e+00, math.sqrt(2) / 2, math.sqrt(2) / 2]
        ])
        trVect = np.array([
            [1.000000e+00],
            [1.000000e+00],
            [1.000000e+00]
        ])
        p1 = point1
        p1 = np.array([p1[0] + w / 2, p1[1] + h / 2, z])  # p1 in camera1 basis
        x1, y1, z1 = p1[0], p1[1], p1[2]

        p2 = point2
        p2 = np.array([p2[0] + w / 2, p2[1] + h / 2, z])  # p2 in camera2 basis
        p2 = np.dot(self.rotMat, p2)  # p2 in camera1 basis
        x2, y2, z2 = p2[0], p2[1], p2[2]

        tx, ty, tz = self.trVect[0], self.trVect[1], self.trVect[2]  # translation vector

        S1 = (ty * x1 - tx * y1) / (x2 * y1 - y2 * x1)
        S2 = (tz * x1 - tx * z1) / (x2 * z1 - z2 * x1)
        T1 = (tx + S1 * x2) / x1
        T2 = (tx + S2 * x2) / x1
        x0, y0, z0 = [0.0], [0.0], [0.0]
        if (np.all(abs((T1 * x1) - (tx + S1 * x2))) < MagicConstants.reconstructEPS):
            x0 = T1 * x1
        if (np.all(abs((T1 * y1) - (ty + S1 * y2))) < MagicConstants.reconstructEPS):
            y0 = T1 * x1
        if (np.all(abs((T1 * z1) - (tz + S1 * z2))) < MagicConstants.reconstructEPS):
            z0 = T1 * x1
        if (np.all(abs((T2 * x1) - (tx + S2 * x2))) < MagicConstants.reconstructEPS):
            x0 = T2 * x1
        if (np.all(abs((T2 * y1) - (ty + S2 * y2))) < MagicConstants.reconstructEPS):
            y0 = T2 * x1
        if (np.all(abs((T2 * z1) - (tz + S2 * z2))) < MagicConstants.reconstructEPS):
            z0 = T2 * x1
        return [x0, y0, z0]

    def reconstructPoint(self, point1, point2):
        z = self.cam.focal
        w, h = self.cam.imgsize
        p1 = point1
        p1 = np.array([p1[0] + w / 2, p1[1] + h / 2, z])  # p1 in camera1 basis
        x1, y1, z1 = p1[0], p1[1], p1[2]

        p2 = point2
        p2 = np.array([p2[0] + w / 2, p2[1] + h / 2, z])  # p2 in camera2 basis
        p2 = np.dot(self.rotMat, p2)  # p2 in camera1 basis
        x2, y2, z2 = p2[0], p2[1], p2[2]

        tx, ty, tz = self.trVect[0], self.trVect[1], self.trVect[2]  # translation vector

        S1 = (ty * x1 - tx * y1) / (x2 * y1 - y2 * x1)
        S2 = (tz * x1 - tx * z1) / (x2 * z1 - z2 * x1)
        T1 = (tx + S1 * x2) / x1
        T2 = (tx + S2 * x2) / x1
        x0, y0, z0 = [0.0], [0.0], [0.0]
        if (np.all(abs((T1 * x1) - (tx + S1 * x2))) < MagicConstants.reconstructEPS):
            x0 = T1 * x1
        if (np.all(abs((T1 * y1) - (ty + S1 * y2))) < MagicConstants.reconstructEPS):
            y0 = T1 * x1
        if (np.all(abs((T1 * z1) - (tz + S1 * z2))) < MagicConstants.reconstructEPS):
            z0 = T1 * x1
        if (np.all(abs((T2 * x1) - (tx + S2 * x2))) < MagicConstants.reconstructEPS):
            x0 = T2 * x1
        if (np.all(abs((T2 * y1) - (ty + S2 * y2))) < MagicConstants.reconstructEPS):
            y0 = T2 * x1
        if (np.all(abs((T2 * z1) - (tz + S2 * z2))) < MagicConstants.reconstructEPS):
            z0 = T2 * x1
        return [x0[0], y0[0], z0[0]]

    def pointCloud(self, pts1, pts2):
        EPS = 0.001
        points3Dx = []
        points3Dy = []
        points3Dz = []
        pts3Ds = []
        z = self.cam.focal
        w, h = self.cam.imgsize
        for i in range(0, len(pts1)):
            point3D = self.reconstructPoint(pts1[i], pts2[i])
            points3Dx.append(point3D[0])
            points3Dy.append(point3D[1])
            points3Dz.append(point3D[2])
            pts3Ds.append([point3D[0], point3D[1], point3D[2]])
        # pts3Ds = np.array(pts3Ds)
        point, normal = self.planeFitRansac(pts3Ds, False)
        equation = self.planeFitRansac(pts3Ds)
        # print(plane)
        #plot3Dcloud(points3Dx, points3Dy, points3Dz, point, normal)
        return equation

    def pixelOnRoad(self, botimg1, botimg2, plane):
        road_pts1 = []
        road_pts2 = []
        height, width, depth = botimg1.shape
        bot1gr = cv2.cvtColor(botimg1, cv2.COLOR_BGR2GRAY)
        bot2gr = cv2.cvtColor(botimg2, cv2.COLOR_BGR2GRAY)

        #print(botimg1.shape)
        #print(botimg2.shape)
        flow = cv2.calcOpticalFlowFarneback(bot1gr, bot2gr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow)
        pts1 = []
        pts2 = []
        for i in range(8, height-8, 8):
            for j in range(8, width-8, 8):
                # было по другому -- j i !!
                point2 = [j+0.0, i+0.0]
                pts2.append(point2)
                #print(flow[i][j])
                point1 = [j-flow[i][j][0]+0.0, i-flow[i][j][1]+0.0]
                pts1.append(point1)
                point3D = self.reconstructPoint(point1, point2)
                A, B, C, D = plane[0], plane[1], plane[2], plane[3]
                #print("A="+str(A) + "B="+str(B)+"C="+str(C)+"D="+str(D))
                #print(point3D)
                x, y, z = point3D[0], point3D[1], point3D[2]
                #print("x="+str(x)+"y="+str(y)+"z="+str(z))
                #print(plane)
                #print(point3D[0])
                plane = [A, B, C, D]
                #point3D = point3D.tolist()
                ev = self.distPlanePt(plane, point3D)
                    #ev = A * x[0] + B * y[0] + C * z[0] + D
                #print(ev)
                if (abs(ev) <= MagicConstants.distToRoad):
                    #print(ev)
                    road_pts1.append(point1)
                    road_pts2.append(point2)
        drw = Drawer(botimg1, botimg2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        #drw.drawFlow(pts1, pts2, 'outout.jpg', botimg2, True)
        road_pts1 = np.int32(road_pts1)
        road_pts2 = np.int32(road_pts2)
        return road_pts1, road_pts2

        '''
        road_pts1 = []
        road_pts2 = []
        height, width, depth = botimg1.shape
        bot1gr = cv2.cvtColor(botimg1, cv2.COLOR_BGR2GRAY)
        bot2gr = cv2.cvtColor(botimg2, cv2.COLOR_BGR2GRAY)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        for point1, point2 in zip(bot1gr, bot2gr):
            point1 = [point1[0], point1[1]]
            point2 = [point2[0], point2[1]]
            #print(point1)
            point3D = self.reconstructPoint(point1, point2)
            A, B, C, D = plane[0], plane[1], plane[2], plane[3]
            x, y, z = point3D[0], point3D[1], point3D[2]
            if (np.any(A * x + B * y + C * z + D) < 6):
                #print("kek")
                road_pts1.append(point1)
                road_pts2.append(point2)
        return road_pts1, road_pts2
        #sel = PointSelector(botimg1, botimg2, (width, height), 80, True)
        #road1, road2 = sel.lucasKanade()
        #drw = Drawer(botimg1, botimg2)
        #drw.drawFlow(road1, road2, 'bottom.jpg', botimg1, True)
        #for point1, point2 in zip(road1, road2):
        #points1 = []
        #for i in range(4, height-4, 4):
        #    for j in range(4, width-4, 4):
        #        points1.append([[j + 0.0,i + 0.0]])
        #points1 = np.array(points1)
        #print(points1)
        '''
        '''
        print(bot1gr.shape)
        print(bot2gr.shape)
        flow = cv2.calcOpticalFlowFarneback(bot1gr, bot2gr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow)
        for i in range(4, height-4, 4):
            for j in range(4, width-4, 4):
                point2 = [i+0.0, j+0.0]
                print(flow[i][j])
                point1 = [i-flow[i][j]+0.0, j-flow[i][j]+0.0]
                point3D = self.reconstructPoint(point1, point2)
                A, B, C, D = plane[0], plane[1], plane[2], plane[3]
                x, y, z = point3D[0], point3D[1], point3D[2]
                if (np.any(A * x + B * y + C * z + D) < 3):
                    print(point2)
                    print(point1)
                    road_pts1.append(point1)
                    road_pts2.append(point2)
        return road_pts1, road_pts2
        '''

