import math
from vec2d import Vec2d

K_MATH_EPS = 1e-10


def cross_prod(point1, point2, point3):
    return (point2 - point1).cross_prod(point3 - point1)


def inner_prod(point1, point2, point3):
    return (point2 - point1).inner_prod(point3 - point1)


def is_within(val, bound1, bound2):
    if bound1 > bound2:
        bound1, bound2 = bound2, bound1
    return val >= bound1 - K_MATH_EPS and val <= bound2 + K_MATH_EPS


class Segment2d:

    def __init__(self, start=None, end=None):
        if start is None and end is None:
            self._start = Vec2d(1, 0)
            self._end = Vec2d(1, 0)
            self._unit_direction = Vec2d(1, 0)
            self._length = 0.0
            self._heading = 0.0
        else:
            self._start = start
            self._end = end
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            self._length = math.hypot(dx, dy)
            if self._length <= K_MATH_EPS:
                self._unit_direction = Vec2d(0, 0)
            else:
                self._unit_direction = Vec2d(dx / self._length, dy / self._length)
            self._heading = self._unit_direction.angle()

    def start(self):
        return self._start

    def end(self):
        return self._end

    def unit_direction(self):
        return self._unit_direction

    def heading(self):
        return self._heading

    def length(self):
        return self._length

    def length_sqr(self):
        return self._length * self._length

    def distance_to(self, point):
        if self._length <= K_MATH_EPS:
            return self._start.distance_to(point)
        proj = self.project_onto_unit(point)
        if proj <= 0.0:
            return self._start.distance_to(point)
        if proj >= self._length:
            return self._end.distance_to(point)
        return abs(self.product_onto_unit(point))

    def distance_to_with_nearest_point(self, point):
        if self._length <= K_MATH_EPS:
            return point.distance_to(self._start), self._start
        proj = self.project_onto_unit(point)
        if proj <= 0.0:
            return self._start.distance_to(point), self._start
        if proj >= self._length:
            return self._end.distance_to(point), self._end
        return self.get_perpendicular_foot(point)

    def distance_sqr_to(self, point):
        return self.distance_to(point) ** 2

    def distance_sqr_to_with_nearest_point(self, point):
        distance, nearest_point = self.distance_to_with_nearest_point(point)
        return distance**2, nearest_point

    def is_point_in(self, point):
        if self._length <= K_MATH_EPS:
            return self._start == point
        prod = cross_prod(point, self._start, self._end)
        if abs(prod) > K_MATH_EPS:
            return False
        return is_within(point.x(), self._start.x(), self._end.x()) and is_within(
            point.y(), self._start.y(), self._end.y()
        )

    def project_onto_unit(self, point):
        # a * cos(theta) = a * b / |b|
        return self._unit_direction.inner_prod(point - self._start)

    def product_onto_unit(self, point):
        # a * sin(theta) = a * b / |b|
        return self._unit_direction.cross_prod(point - self._start)

    def has_intersect(self, other_segment):
        return self.get_intersect(other_segment) is not None

    def get_intersect(self, other_segment):
        if self.is_point_in(other_segment.start()):
            return other_segment.start()
        if self.is_point_in(other_segment.end()):
            return other_segment.end()
        if other_segment.is_point_in(self._start):
            return self._start
        if other_segment.is_point_in(self._end):
            return self._end
        if self._length <= K_MATH_EPS or other_segment.length() <= K_MATH_EPS:
            return None

        cc1 = cross_prod(self._start, self._end, other_segment.start())
        cc2 = cross_prod(self._start, self._end, other_segment.end())
        if cc1 * cc2 >= -K_MATH_EPS:
            return None
        cc3 = cross_prod(other_segment.start(), other_segment.end(), self._start)
        cc4 = cross_prod(other_segment.start(), other_segment.end(), self._end)
        if cc3 * cc4 >= -K_MATH_EPS:
            return None
        # print(f"cc1: {cc1:.3f}, cc2: {cc2:.3f}, cc3: {cc3:.3f}, cc4: {cc4:.3f}")
        ratio = cc4 / (cc4 - cc3)
        return Vec2d(
            self._start.x() * ratio + self._end.x() * (1.0 - ratio),
            self._start.y() * ratio + self._end.y() * (1.0 - ratio),
        )

    def get_perpendicular_foot(self, point):
        if self._length <= K_MATH_EPS:
            return point.distance_to(self._start), self._start
        proj = self.project_onto_unit(point)
        foot_point = self._start + self._unit_direction * proj
        return abs(self.product_onto_unit(point)), foot_point

    def debug_string(self):
        return (
            "segment2d ("
            + self._start.debug_string()
            + ", "
            + self._end.debug_string()
            + ")"
        )


def test1():
    start = Vec2d(0, 0)
    end = Vec2d(5, 5)
    line = Segment2d(start, end)

    points = [Vec2d(4, 3), Vec2d(6, 8), Vec2d(-1, 1)]
    for idx, pt in enumerate(points, 1):
        print(f"Vec2d {idx}: ({pt})")

        distance = line.distance_to(pt)
        print(f"Distance to line segment: {distance:.3f}")

        distance, nearest_point = line.distance_to_with_nearest_point(pt)
        print(f"Distance to line segment (with nearest point): {distance:.3f}")
        print(f"Nearest point on line segment: ({nearest_point})\n")

        distance_sqr, nearest_point = line.distance_sqr_to_with_nearest_point(pt)
        print(f"Distance sqr to line segment (with nearest point): {distance_sqr:.3f}")
        print(f"Nearest point on line segment: ({nearest_point})\n")

        distance, foot_point = line.get_perpendicular_foot(pt)
        print(f"Distance to line segment (with nearest point): {distance:.3f}")
        print(f"Foot point on line segment: ({foot_point})\n")


def test2():
    # Test 1: Intersection of two line segments
    A = Vec2d(1, 1)
    B = Vec2d(5, 5)
    C = Vec2d(1, 5)
    D = Vec2d(5, 1)
    line1 = Segment2d(A, B)
    line2 = Segment2d(C, D)
    intersection = line1.get_intersect(line2)

    print(f"Test 1 - Intersection: {intersection}, {line1.has_intersect(line2)}")

    # Test 2: No intersection of two line segments
    A2 = Vec2d(1, 1)
    B2 = Vec2d(3, 1)
    C2 = Vec2d(5, 5)
    D2 = Vec2d(6, 6)
    line3 = Segment2d(A2, B2)
    line4 = Segment2d(C2, D2)
    intersection2 = line3.get_intersect(line4)

    print(f"Test 2 - Intersection: {intersection2}, {line3.has_intersect(line4)}")


if __name__ == "__main__":
    test1()
    # test2()
