import math
from vec2d import Vec2d

K_MATH_EPS = 1e-10


class AABox2d:
    def __init__(
        self,
        params=None,
        corners=None,
        points=None,
    ):
        if params:
            if isinstance(params, tuple) and len(params) == 3:
                center, length, width = params
                self._center = center
                self._length = length
                self._width = width
                self._half_length = length / 2.0
                self._half_width = width / 2.0
            else:
                raise ValueError("params must be a tuple of length 3")
        elif corners:
            if isinstance(corners, tuple) and len(corners) == 2:
                one_corner, opponent_corner = corners
                center_x = (one_corner.x() + opponent_corner.x()) / 2.0
                center_y = (one_corner.y() + opponent_corner.y()) / 2.0
                self._center = Vec2d(center_x, center_y)
                self._length = abs(one_corner.x() - opponent_corner.x())
                self._width = abs(one_corner.y() - opponent_corner.y())
                self._half_length = self._length / 2.0
                self._half_width = self._width / 2.0
            else:
                raise ValueError("corners must be a tuple of length 2")
        elif points:
            if isinstance(points, tuple) and len(points) == 1:
                points = points[0]
                min_x = points[0].x()
                max_x = points[0].x()
                min_y = points[0].y()
                max_y = points[0].y()
                for point in points:
                    min_x = min(min_x, point.x())
                    max_x = max(max_x, point.x())
                    min_y = min(min_y, point.y())
                    max_y = max(max_y, point.y())
                self._center = Vec2d((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                self._length = max_x - min_x
                self._width = max_y - min_y
                self._half_length = self._length / 2.0
                self._half_width = self._width / 2.0
            else:
                raise ValueError("points must be a tuple of length 1")
        else:
            raise ValueError("At least one parameter must be provided")

    def center(self):
        return self._center

    def center_x(self):
        return self._center.x()

    def center_y(self):
        return self._center.y()

    def length(self):
        return self._length

    def width(self):
        return self._width

    def half_length(self):
        return self._half_length

    def half_width(self):
        return self._half_width

    def area(self):
        return self._length * self._width

    def min_x(self):
        return self._center.x() - self._half_length

    def max_x(self):
        return self._center.x() + self._half_length

    def min_y(self):
        return self._center.y() - self._half_width

    def max_y(self):
        return self._center.y() + self._half_width

    def get_all_corners(self):
        rb = Vec2d(
            self._center.x() + self._half_length,
            self._center.y() - self._half_width,
        )
        rt = Vec2d(
            self._center.x() + self._half_length,
            self._center.y() + self._half_width,
        )
        lt = Vec2d(
            self._center.x() - self._half_length,
            self._center.y() + self._half_width,
        )
        lb = Vec2d(
            self._center.x() - self._half_length,
            self._center.y() - self._half_width,
        )
        return [rb, rt, lt, lb]

    def is_point_in(self, point):
        vec = Vec2d(point.x() - self._center.x(), point.y() - self._center.y())
        return (
            abs(vec.x()) <= self._half_length + K_MATH_EPS
            and abs(vec.y()) <= self._half_width + K_MATH_EPS
        )

    def is_point_on_boundary(self, point):
        dx = abs(point.x() - self._center.x())
        dy = abs(point.y() - self._center.y())
        return (
            abs(dx - self._half_length) <= K_MATH_EPS
            and dy <= self._half_width + K_MATH_EPS
        ) or (
            abs(dy - self._half_width) <= K_MATH_EPS
            and dx <= self._half_length + K_MATH_EPS
        )

    def distance_to_point(self, point):
        dx = abs(point.x() - self._center.x()) - self._half_length
        dy = abs(point.y() - self._center.y()) - self._half_width
        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return math.sqrt(dx**2 + dy**2)

    def distance_to_aabox(self, box):
        dx = (
            abs(box.center_x() - self._center.x())
            - box.half_length()
            - self._half_length
        )
        dy = (
            abs(box.center_y() - self._center.y()) - box.half_width() - self._half_width
        )
        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return math.sqrt(dx**2 + dy**2)

    def has_overlap(self, box):
        return (
            abs(box.center_x() - self._center.x())
            <= box.half_length() + self._half_length + K_MATH_EPS
        ) and (
            abs(box.center_y() - self._center.y())
            <= box.half_width() + self._half_width + K_MATH_EPS
        )

    def shift(self, shift_vec):
        print("shift_vec:", self._center.debug_string())
        self._center += shift_vec

    def merge_from_aabox(self, other_box):
        x1 = min(self.min_x(), other_box.min_x())
        x2 = max(self.max_x(), other_box.max_x())
        y1 = min(self.min_y(), other_box.min_y())
        y2 = max(self.max_y(), other_box.max_y())
        self._center = Vec2d((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self._length = x2 - x1
        self._width = y2 - y1
        self._half_length = self._length / 2.0
        self._half_width = self._width / 2.0

    def merge_from_point(self, other_point):
        x1 = min(self.min_x(), other_point.x())
        x2 = max(self.max_x(), other_point.x())
        y1 = min(self.min_y(), other_point.y())
        y2 = max(self.max_y(), other_point.y())
        self._center = Vec2d((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self._length = x2 - x1
        self._width = y2 - y1
        self._half_length = self._length / 2.0
        self._half_width = self._width / 2.0

    def debug_string(self):
        return f"aabox2d ( center = {self._center.debug_string()}  length = {self._length}  width = {self._width}  points = ({self.min_x()}, {self.min_y()}) ({self.min_x()}, {self.max_y()}) ({self.max_x()}, {self.max_y()}) ({self.max_x()}, {self.min_y()}))"


def test1():
    aabb1 = AABox2d(params=(Vec2d(0, 0), 10, 20))
    print("AABox2d by params:", aabb1.debug_string())

    corner1 = Vec2d(0, 0)
    corner2 = Vec2d(10, 20)
    aabb2 = AABox2d(corners=(corner1, corner2))
    print("AABox2d by corners:", aabb2.debug_string())

    points = [Vec2d(0, 0), Vec2d(5, 10), Vec2d(10, 5)]
    aabb3 = AABox2d(points=(points,))
    print("AABox2d by points:", aabb3.debug_string())


def test2():
    aabox1 = AABox2d(params=(Vec2d(0, 0), 4, 6))
    print("AABox2d by params:", aabox1.debug_string())
    aabox2 = AABox2d(params=(Vec2d(5, 0), 4, 6))
    print("AABox2d by params:", aabox2.debug_string())
    point = Vec2d(1, 1)

    # Test if a point is inside a box
    print(f"Point {point.debug_string()} inside aabox1: {aabox1.is_point_in(point)}")

    # Test if a point is on the boundary of a box
    point_on_boundary = Vec2d(2, 3)
    print(
        f"Point {point_on_boundary.debug_string()} on boundary of aabox1: {aabox1.is_point_on_boundary(point_on_boundary)}"
    )

    # Test distance from point to box
    print(
        f"Distance from {point.debug_string()} to aabox1: {aabox1.distance_to_point(point)}"
    )

    # Test distance between two boxes
    print(f"Distance between aabox1 and aabox2: {aabox1.distance_to_aabox(aabox2)}")

    # Test if boxes overlap
    print(f"Do aabox1 and aabox2 overlap? {aabox1.has_overlap(aabox2)}")

    # Shift aabox1
    aabox1.shift(Vec2d(2, 0))
    print(f"New center of aabox1: {aabox1._center.debug_string()}")

    # Merge aabox1 with aabox2
    aabox1.merge_from_aabox(aabox2)
    print(f"New center of merged aabox1: {aabox1._center.debug_string()}")

    # Merge aabox1 with a point
    aabox1.merge_from_point(Vec2d(6, 2))
    print(f"New center of merged aabox1: {aabox1._center.debug_string()}")


if __name__ == "__main__":
    # test1()
    test2()
