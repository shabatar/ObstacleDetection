def unionRects(rects1, rects2):
    union = []
    for rect1 in rects1:
        for rect2 in rects2:
            if rect1 == rect2:
                continue
            if (rect1.isCloseTo(rect2)):
                u = rect1.union(rect2)
                if (u.large()):
                    union.append(u)
    if (len(union) == 0):
        return rects1
    return union

class Rectangle:
    def __init__(self, minX, maxY, maxX, minY):
        self.minX = minX
        self.maxY = maxY
        self.maxX = maxX
        self.minY = minY

    def getCorners(self):
        return ([self.minX, self.minY], [self.maxX, self.minY], [self.minX, self.maxY], [self.maxX, self.maxY])

    def isPointInside(self, point):
        return (self.minX < point[0] < self.maxX) and (self.minY < point[1] < self.maxY)

    def isCloseTo(self, otherRect):
        """
        :type otherRect: Rectangle
        """
        corners = self.getCorners()
        return (otherRect.isPointInside(corners[0]) or
                otherRect.isPointInside(corners[1]) or
                otherRect.isPointInside(corners[2]) or
                otherRect.isPointInside(corners[3])
        )
    def large(self):
        S = (self.maxX - self.minX) * (self.maxY - self.minY)
        return (S > 50)

    def union(self, rect):
        minX1, maxY1, maxX1, minY1 = self.minX, self.maxY, self.maxX, self.maxY
        minX2, maxY2, maxX2, minY2 = rect.minX, rect.maxY, rect.maxX, rect.maxY
        return Rectangle(min(minX1, minX2), max(maxY1, maxY2), max(maxX1, maxX2), min(minY1, minY2))