class Camera:
    def __init__(self, calibMat, distCoeffs, focal, pp, imgsize):
        self.calibMat = calibMat
        self.distCoeffs = distCoeffs
        self.focal = focal
        self.pp = pp
        self.imgsize = imgsize