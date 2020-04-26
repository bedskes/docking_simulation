import numpy as np

class Path:
    def __init__(self, points):
        #for now limit to 2 points, should revisit later for multiple waypoints
        points = np.array(points)
        self.points = points
        self.x_vals = [pt[0] for pt in points]
        self.y_vals = [pt[1] for pt in points]
        diff = points[1] - points[0]
        self.direction = np.arctan2(diff[1], diff[0])

    def distance_to(self, point):
        # for line x = a+t*n where n is the direction,
        # d = |(a-p) - ((a-p).n).n|
        diff = (self.points[0] - point)
        d = np.linalg.norm(diff - np.dot(np.dot(diff, self.direction), self.direction))
        return d

    def angle_to(self, point):
        diff = (point - self.points[0])
        return np.arctan2(diff[1], diff[0]) - self.direction
