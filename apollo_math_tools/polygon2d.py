import math
from vec2d import Vec2d
from segment2d import Segment2d
from aabox2d import AABox2d
from box2d import Box2d
from util import cross_prod, normalize_angle


K_MATH_EPS = 1e-10


class Polygon2d:
    def __init__(self, points=None):
        if isinstance(points, Box2d):
            self._points = points.get_all_corners()
        elif isinstance(points, list):
            self._points = points
        else:
            self._points = []
        self.num_points_ = len(self._points)
        self._segments = []
        self._is_convex = False
        self._area = 0.0
        self._min_x = float("inf")
        self._max_x = float("-inf")
        self._min_y = float("inf")
        self._max_y = float("-inf")
        self.build_from_points()

    def points(self):
        return self._points

    def num_points(self):
        return self.num_points_

    def segments(self):
        return self._segments

    def is_convex(self):
        return self._is_convex

    def area(self):
        return self._area

    def min_x(self):
        return self._min_x

    def max_x(self):
        return self._max_x

    def min_y(self):
        return self._min_y

    def max_y(self):
        return self._max_y

    def next(self, at):
        return 0 if at >= self.num_points_ - 1 else at + 1

    def prev(self, at):
        return self.num_points_ - 1 if at == 0 else at - 1

    def build_from_points(self):
        # Construct segments.
        self._segments.clear()
        for i in range(self.num_points_):
            self._segments.append(
                Segment2d((self._points[i], self._points[self.next(i)]))
            )
        if self.num_points_ < 3:
            print(f"num_points_({self.num_points_}) < 3, crash error")
            return

        # Make sure the points are in counter-clockwise order.
        self._area = 0.0
        for i in range(1, self.num_points_):
            self._area += cross_prod(
                self._points[0], self._points[i - 1], self._points[i]
            )
        if self._area < 0:
            self._area = -self._area
            self._points.reverse()
        self._area /= 2.0
        if self._area <= K_MATH_EPS:
            print("polygon area is too tiny, crash error")
            return

        # Check convexity.
        self.is_convex_ = True
        for i in range(self.num_points_):
            if (
                cross_prod(
                    self._points[self.prev(i)],
                    self._points[i],
                    self._points[self.next(i)],
                )
                <= -K_MATH_EPS
            ):
                self.is_convex_ = False
                break

        # Compute aabox.
        self._min_x = self._points[0].x()
        self._max_x = self._points[0].x()
        self._min_y = self._points[0].y()
        self._max_y = self._points[0].y()
        for point in self._points:
            self._min_x = min(self._min_x, point.x())
            self._max_x = max(self._max_x, point.x())
            self._min_y = min(self._min_y, point.y())
            self._max_y = max(self._max_y, point.y())

    def aa_bounding_box(self):
        return AABox2d(
            corners=(Vec2d(self._min_x, self._min_y), Vec2d(self._max_x, self._max_y))
        )

    def is_point_on_boundary(self, point):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return True
        return any(poly_seg.is_point_in(point) for poly_seg in self._segments)

    def is_point_in(self, point):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return True
        if self.is_point_on_boundary(point):
            return True
        c = 0
        j = self.num_points_ - 1
        for i in range(self.num_points_):
            if (self._points[i].y() > point.y()) != (self._points[j].y() > point.y()):
                side = cross_prod(point, self._points[i], self._points[j])
                if self._points[i].y() < self._points[j].y():
                    if side > 0:
                        c += 1
                else:
                    if side < 0:
                        c += 1
            j = i
        return c & 1

    def distance_to_boundary(self, point):
        distance = float("inf")
        for poly_seg in self._segments:
            distance = min(distance, poly_seg.distance_to(point))
        return distance

    def distance_to_point(self, point):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return 0.0
        if self.is_point_in(point):
            return 0.0
        return self.distance_to_boundary(point)

    def distance_to_segment(self, segment):
        if segment.length() <= K_MATH_EPS:
            return self.distance_to_point(segment.start())
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return 0.0
        if self.is_point_in(segment.center()):
            return 0.0
        if any(segment.has_intersect(poly_seg) for poly_seg in self._segments):
            return 0.0

        distance = min(
            self.distance_to_point(segment.start()),
            self.distance_to_point(segment.end()),
        )
        for i in range(self.num_points_):
            distance = min(distance, segment.distance_to(self._points[i]))
        return distance

    def distance_to_polygon(self, polygon):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return 0.0
        if polygon.num_points() < 3:
            print(f"polygon.size({polygon.num_points()}) < 3, crash error")
            return 0.0
        if self.is_point_in(polygon.points()[0]):
            return 0.0
        if polygon.is_point_in(self._points[0]):
            return 0.0
        distance = float("inf")
        for i in range(self.num_points_):
            distance = min(distance, polygon.distance_to_segment(self._segments[i]))
        return distance

    def distance_to_box(self, box):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return 0.0
        return self.distance_to_polygon(Polygon2d(box))

    def has_overlap_polygon(self, polygon):
        if self.num_points_ < 3:
            print(f"points_.size({self.num_points_}) < 3, crash error")
            return True
        self_aabox = self.aa_bounding_box()
        traget_aabox = polygon.aa_bounding_box()
        if not self_aabox.has_overlap(traget_aabox):
            return False
        return self.distance_to_polygon(polygon) <= K_MATH_EPS

    @staticmethod
    def compute_convex_hull(points):
        n = len(points)
        if n < 3:
            print("points.size({}) < 3".format(n))
            return False, None
        # Step 1: Sort the points
        sorted_indices = list(range(n))
        sorted_indices.sort(key=lambda idx: (points[idx].x(), points[idx].y()))

        # Step 2: Compute the convex hull: Monotone Chain Algorithm
        lower = []
        for i in range(n):
            while (
                len(lower) >= 2
                and cross_prod(
                    points[lower[-2]], points[lower[-1]], points[sorted_indices[i]]
                )
                <= K_MATH_EPS
            ):
                lower.pop()
            lower.append(sorted_indices[i])

        upper = []
        for i in range(n - 1, -1, -1):
            while (
                len(upper) >= 2
                and cross_prod(
                    points[upper[-2]], points[upper[-1]], points[sorted_indices[i]]
                )
                <= K_MATH_EPS
            ):
                upper.pop()
            upper.append(sorted_indices[i])

        # Remove the last point of each half because it's repeated at the beginning of the other half
        del lower[-1]
        del upper[-1]

        convex_hull_indices = lower + upper
        convex_hull_points = [points[idx] for idx in convex_hull_indices]
        return True, Polygon2d(convex_hull_points)


def test():
    import matplotlib.pyplot as plt
    import random

    def generate_random_points(num_points, x_range, y_range):
        return [
            Vec2d(
                random.uniform(x_range[0], x_range[1]),
                random.uniform(y_range[0], y_range[1]),
            )
            for _ in range(num_points)
        ]

    def plot_convex_hull(points, polygon):
        x_points = [p.x() for p in points]
        y_points = [p.y() for p in points]
        plt.scatter(x_points, y_points, color="blue", label="Points")

        if polygon:
            polygon_x = [p.x() for p in polygon.points()]
            polygon_y = [p.y() for p in polygon.points()]
            plt.plot(
                polygon_x + [polygon_x[0]],
                polygon_y + [polygon_y[0]],
                color="red",
                label="Convex Hull",
            )
        plt.legend()
        plt.show()

    num_points = 12
    x_range = (0, 10)
    y_range = (0, 10)
    points = generate_random_points(num_points, x_range, y_range)
    success, polygon = Polygon2d.compute_convex_hull(points)
    if not success:
        print("No Convex Hull found")
    plot_convex_hull(points, polygon)


if __name__ == "__main__":
    test()
