import cv2
import numpy as np
from constants import MagicConstants
from pointselector import PointSelector
from drawer import Drawer
cnst = MagicConstants()
class Reconstructor:
    def __init__(self, cam, rotMat, trVect):
        self.cam = cam
        self.rotMat = rotMat
        self.trVect = trVect

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

    def bestFitPlane(self, points, equation=True):
        w, v = self.PCA(points)
        import random
        points = random.sample(points, cnst.roadPlanePts)
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
        if (np.all(abs((T1 * x1) - (tx + S1 * x2))) < cnst.reconstructEPS):
            x0 = T1 * x1
        if (np.all(abs((T1 * y1) - (ty + S1 * y2))) < cnst.reconstructEPS):
            y0 = T1 * x1
        if (np.all(abs((T1 * z1) - (tz + S1 * z2))) < cnst.reconstructEPS):
            z0 = T1 * x1
        if (np.all(abs((T2 * x1) - (tx + S2 * x2))) < cnst.reconstructEPS):
            x0 = T2 * x1
        if (np.all(abs((T2 * y1) - (ty + S2 * y2))) < cnst.reconstructEPS):
            y0 = T2 * x1
        if (np.all(abs((T2 * z1) - (tz + S2 * z2))) < cnst.reconstructEPS):
            z0 = T2 * x1
        return [x0, y0, z0]

    def pointCloud(self, pts1, pts2):
        EPS = 0.001
        points3Dx = []
        points3Dy = []
        points3Dz = []
        pts3Ds = []
        z = self.cam.focal
        w, h = self.cam.imgsize
        for i in range(0, len(pts1)):
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
        # pts3Ds = np.array(pts3Ds)
        point, normal = self.bestFitPlane(pts3Ds, False)
        equation = self.bestFitPlane(pts3Ds)
        # print(plane)
        #self.plot3Dcloud(points3Dx,points3Dy,points3Dz, point, normal)
        return equation

    def plot3Dcloud(self, points3Dx, points3Dy, points3Dz, point, normal):
        import matplotlib.pyplot as plt
        xmin = min(points3Dx)[0]
        ymin = min(points3Dy)[0]
        zmin = min(points3Dz)[0]
        zmax = max(points3Dz)[0]
        xmax = max(points3Dx)[0]
        ymax = max(points3Dy)[0]
        from mpl_toolkits.mplot3d import Axes3D
        # d and we're set
        d = -point.dot(normal)
        if(xmin == xmax or ymin == ymax or zmin == zmax):
            return 0
        # create x,y
        xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

        # calculate corresponding z
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

        # plot the surface
        plt3d = plt.figure().gca(projection='3d')
        plt3d.plot_surface(xx, yy, z, alpha=0.5)

        ax = plt.gca()
        ax.hold(True)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points3Dx, points3Dy, points3Dz, c='r', marker='*')
        plt.savefig('fig.png')
        print(xmin)
        plt.xlim(xmin, xmax)
        print(ymin)
        plt.ylim(ymin, ymax)
        print(zmin)
        ax.set_zlim(zmin, zmax)
        plt.show()

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
                try:
                    ev = A * x + B * y + C * z + D
                except Exception:
                    continue
                ev = ev[0]
                #print(ev)
                if (abs(ev) < 0.1):
                    #print("kek")
                    #print(abs(ev - 1))
                    #print(ev)
                    #print(ev)
                    #print(point2)
                    #print(point1)
                    road_pts1.append(point1)
                    road_pts2.append(point2)
        drw = Drawer(botimg1, botimg2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        drw.drawFlow(pts1, pts2, 'outout.jpg', botimg2, True)
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
