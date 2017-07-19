
class MagicConstants:
    def __init__(self):
        self.clusterConstant =  70
        self.closeToEpipolar = 2
        self.optFlowMagn = 7
        s = 4
        self.gaussWin = (s, s)
        self.meanShiftN = 16
        self.pointToSelect = 42
        self.subtractGNum = 20
        self.reconstructEPS = 0.00001
        self.largeModuleCnst = 0.3
        self.roadPlanePts = 6
        self.critAlpha = 0.5
        self.minPinClust = 5