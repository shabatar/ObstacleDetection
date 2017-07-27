import cv2
import numpy as np
from constants import MagicConstants
from drawer import *
import random
import math
from visodometry import Geometry
class Reconstructor:
    def __init__(self, cam, rotMat, trVect):
        self.cam = cam
        self.rotMat = rotMat
        self.trVect = trVect

    def inliersCnt(self, points, plane):
        cnt = 0
        for p in points:
            if (abs(plane[0] * p[0] + plane[1]*p[1] + plane[2]*p[2] + plane[3]) < MagicConstants.reconstructEPS):
                cnt += 1
        return cnt

    def planeFitRansac(self, points, equation=True):
        just3pts = random.sample(points, 3)
        best3pts = just3pts
        plane = Geometry.planeByPoints(just3pts[0], just3pts[1], just3pts[2])
        bestCnt = sum(Geometry.distPlanePt(plane, pt) for pt in points)
        #bestCnt = self.inliersCnt(points, plane)
        bestPlane = plane
        for i in range(0, 150):
            just3pts = random.sample(points, 3)
            plane = Geometry.planeByPoints(just3pts[0], just3pts[1], just3pts[2])
            cnt = sum([Geometry.distPlanePt(plane, pt) for pt in points])
            #cnt = self.inliersCnt(points, plane)
            if cnt < bestCnt:
                bestCnt = cnt
                best3pts = just3pts
                bestPlane = plane
        if (equation == False):
            point = random.sample(best3pts, 1)
            normal = bestPlane[0], bestPlane[1], bestPlane[2]
            return point, normal
        #print(bestCnt / len(points))
        print("bestcnt="+str(bestCnt))
        return bestPlane

    def reconstructPoint(self, point1, point2, plot=False):
        z = self.cam.focal
        w, h = self.cam.imgsize
        tx, ty, tz = self.trVect[0], self.trVect[1], self.trVect[2]  # translation vector
        # p1 в базисе первой камеры
        # p[0] ... p[1] так было и +
        p1 = np.array([point1[0] - w / 2, point1[1] - h / 2, z])  # p1 in camera1 basis
        x1, y1, z1 = p1[0], p1[1], p1[2]

        # (0,0,z) - центр первого кадра в базисе первой камеры, считаем центр второго кадра..
        pp2 = np.dot(self.rotMat, np.array([0+tx,0+ty,z+tz]))  # точка центра после переноса
        #cam2 = [pp2[0][0], pp2[1][0], pp2[2][0]-z] # переместили камеру по z-координате от точки центра
        pp2 = [pp2[0][0], pp2[1][0], pp2[2][0]] # numpy-евская херня
        r1 = [-w / 2, -h / 2, z]
        r2 = [-w / 2, h / 2, z]
        r3 = [w / 2, h / 2, z]
        r21 = np.dot(self.rotMat, np.array([r1[0] + tx, r1[1] + ty, r1[2] + tz]))
        r22 = np.dot(self.rotMat, np.array([r2[0] + tx, r2[1] + ty, r2[2] + tz]))
        r23 = np.dot(self.rotMat, np.array([r3[0] + tx, r3[1] + ty, r3[2] + tz]))
        r21 = [r21[0][0], r21[1][0], r21[2][0]]
        r22 = [r22[0][0], r22[1][0], r22[2][0]]
        r23 = [r23[0][0], r23[1][0], r23[2][0]]
        frame2 = Geometry.planeByPoints(r23, r21, r22)
        normal = Geometry.unitVect(np.array([frame2[0], frame2[1], frame2[2]]))
        d = z
        cam2 = [pp2[0] + d * normal[0], pp2[1] + d * normal[1], pp2[2] + d * normal[2]]
        #cam2 = np.dot(self.rotMat, np.array([tx[0], ty[0], tz[0]]))

        # ищем вторую точку в первом базисе
        # p[0] ... p[1] так было и +
        p2 = np.array([point2[0] - w / 2, point2[1] - h / 2, z])  # вторая точка во втором базисе (z изменилось)
        p2 = np.dot(self.rotMat, np.array([p2[0]+tx, p2[1]+ty, p2[2]+tz]))  # вторая точка в первом базисе
        #p2[2] = pp2[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]

        A = np.array([
            [x1, -x2, 0],
            [y1, -y2, 0],
            [z1, -z2, 0]
        ])
        b = np.array([tx, ty, tz])
        point = np.linalg.lstsq(A, b)[0]
        T, S = point[0], point[1]
        x0 = T * x1
        y0 = T * y1
        z0 = T * z1

        if (plot):
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import pyplot as plt
            #print([x0[0], y0[0], z0[0]])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            #r1 = [-w/2, -h/2, z]
            #r2 = [-w/2,  h/2, z]
            #r3 = [w/2,   h/2, z]
            r4 = [w/2,  -h/2, z]
            # img1
            x1ar = [-w/2, -w/2, +w/2, +w/2, -w/2]
            y1ar = [-h/2, +h/2, +h/2, -h/2, -h/2]
            z1ar = [   z,    z,    z,    z,    z]

            #r21 = np.dot(self.rotMat, np.array([r1[0] + tx, r1[1] + ty, r1[2] + tz]))
            #r22 = np.dot(self.rotMat, np.array([r2[0] + tx, r2[1] + ty, r2[2] + tz]))
            #r23 = np.dot(self.rotMat, np.array([r3[0] + tx, r3[1] + ty, r3[2] + tz]))
            r24 = np.dot(self.rotMat, np.array([r4[0] + tx, r4[1] + ty, r4[2] + tz]))
            #r21 = [r21[0][0], r21[1][0], r21[2][0]]
            #r22 = [r22[0][0], r22[1][0], r22[2][0]]
            #r23 = [r23[0][0], r23[1][0], r23[2][0]]
            r24 = [r24[0][0], r24[1][0], r24[2][0]]
            # img2
            x2ar = [r21[0], r22[0], r23[0], r24[0], r21[0]]
            y2ar = [r21[1], r22[1], r23[1], r24[1], r21[1]]
            z2ar = [r21[2], r22[2], r23[2], r24[2], r21[2]]
            #verts = [zip(x, y, z)]
            #ax.add_collection3d(Poly3DCollection(verts))
            #ax.scatter(0, 0, z-, c='g', marker='*')
            #ax.plot([0], [0], [0], c='g')
            ax.plot(x1ar, y1ar, z1ar, c='g')
            ax.plot(x2ar, y2ar, z2ar, c='m')
            ax.scatter(cam2[0], cam2[1], cam2[2], c='m', marker='*')
            ax.scatter(pp2[0], pp2[1], pp2[2], c='m', marker='o')
            ax.scatter(0, 0, z, c='g', marker='*')
            ax.scatter(x2, y2, z2, c='m',marker='o')
            ax.plot([0, x1], [0, y1], [0, z1], c='g',marker='*')
            ax.plot([cam2[0], x2], [cam2[1], y2], [cam2[2], z2], c='m',marker='*')
            ax.plot([0, x0], [0, y0], [0, z0], c='b', marker='*')
            plt.xlim(-750, 750)
            # print(ymin)
            plt.ylim(-200, 200)
            # print(zmin)
            ax.set_zlim(-2, 1000)
            plt.show()
        #print([x0[0], y0[0], z0[0]])
        print(z0[0] > 0)
        return [x0[0], y0[0], z0[0]]

    def pointCloud(self, pts1, pts2, plot=False):
        EPS = 0.001
        points3Dx = []
        points3Dy = []
        points3Dz = []
        pts3Ds = []
        z = self.cam.focal
        w, h = self.cam.imgsize
        for i in range(0, len(pts1)):
            point3D = self.reconstructPoint(pts1[i], pts2[i],plot)
            points3Dx.append(point3D[0])
            points3Dy.append(point3D[1])
            points3Dz.append(point3D[2])
            pts3Ds.append([point3D[0], point3D[1], point3D[2]])
        # pts3Ds = np.array(pts3Ds)
        point, normal = self.planeFitRansac(pts3Ds, False)
        equation = self.planeFitRansac(pts3Ds)
        # print(equation)
        plot3Dcloud(points3Dx, points3Dy, points3Dz, point, normal)
        args = zip(points3Dx, points3Dy, points3Dz)
        return equation, args

    def pixelOnRoad(self, img1, img2, plane):
        road_pts1 = []
        road_pts2 = []
        height, width, depth = img1.shape
        img1gr = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2gr = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(img1gr, img2gr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #print(flow)
        pts1 = []
        pts2 = []
        for i in range(0, width, 4):
            for j in range(0, height, 4):
                # верно!!
                point2 = [i+0.0, j+0.0]
                point1 = [int(i-flow[j][i][0]+0.0), int(j-flow[j][i][1]+0.0)]
                pts1.append(point1)
                pts2.append(point2)
                # ??
                point3D = self.reconstructPoint(point1, point2)
                A, B, C, D = plane[0], plane[1], plane[2], plane[3]
                #print("A="+str(A) + "B="+str(B)+"C="+str(C)+"D="+str(D))
                x, y, z = point3D[0], point3D[1], point3D[2]
                #print("x="+str(x)+"y="+str(y)+"z="+str(z))
                plane = [A, B, C, D]
                #point3D = point3D.tolist()
                ev1 = Geometry.distPlanePt(plane, point3D)
                ev = A * x + B * y + C * z + D
                #print("ev1" + str(ev1))
                #print("ev" + str(ev))
                if (ev == 0.0):
                    print("0.0.0")
                if (abs(ev) < MagicConstants.distToRoad):
                    #print(ev)
                    road_pts1.append(point1)
                    road_pts2.append(point2)
        drw = Drawer(img1, img2)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        #drw.drawFlow(pts1, pts2, 'outout.jpg', img2, True)
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

