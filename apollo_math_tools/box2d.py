from math import cos, sin, hypot, pi
from vec2d import Vec2d
from segment2d import Segment2d
from aabox2d import AABox2d

K_MATH_EPS = 1e-10


def wrap_angle(angle):
    new_angle = angle % (2 * pi)
    return new_angle + 2 * pi if new_angle < 0 else new_angle


def normalize_angle(angle):
    new_angle = angle % (2 * pi)
    return (new_angle + 2 * pi if new_angle < 0 else new_angle) - pi


def angle_diff(from_angle, to_angle):
    return normalize_angle(to_angle - from_angle)


class Box2d:
    def __init__(self, center, heading, length, width):
        self._center = center
        self._length = length
        self._width = width
        self._half_length = length / 2.0
        self._half_width = width / 2.0
        self._heading = heading
        self._cos_heading = cos(heading)
        self._sin_heading = sin(heading)
        self._corners = self.init_corners()

    @staticmethod
    def create_aabox(one_corner, opposite_corner):
        x1 = min(one_corner.x(), opposite_corner.x())
        x2 = max(one_corner.x(), opposite_corner.x())
        y1 = min(one_corner.y(), opposite_corner.y())
        y2 = max(one_corner.y(), opposite_corner.y())
        return Box2d(Vec2d((x1 + x2) / 2.0, (y1 + y2) / 2.0), 0.0, x2 - x1, y2 - y1)

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

    def heading(self):
        return self._heading

    def cos_heading(self):
        return self._cos_heading

    def sin_heading(self):
        return self._sin_heading

    def area(self):
        return self._length * self._width

    def diagonal(self):
        return hypot(self._length, self._width)

    def init_corners(self):
        corners_offsets = [
            Vec2d(self._half_length, self._half_width),
            Vec2d(self._half_length, -self._half_width),
            Vec2d(-self._half_length, -self._half_width),
            Vec2d(-self._half_length, self._half_width),
        ]
        corners = []
        for offset in corners_offsets:
            vec = offset.rotate_clockwise(self._heading)
            corner = self._center + vec
            corners.append(corner)
        return corners

    def get_all_corners(self):
        return self._corners

    def is_point_in(self, point):
        offset = Vec2d(point.x() - self._center.x(), point.y() - self._center.y())
        vec = offset.rotate_clockwise(self._heading)
        return (
            abs(vec.x()) <= self._half_length + K_MATH_EPS
            and abs(vec.y()) <= self._half_width + K_MATH_EPS
        )

    def is_point_on_boundary(self, point):
        offset = Vec2d(point.x() - self._center.x(), point.y() - self._center.y())
        vec = offset.rotate_clockwise(self._heading)
        dx = abs(vec.x())
        dy = abs(vec.y())
        return (
            abs(dx - self._half_length) <= K_MATH_EPS
            and dy <= self._half_width + K_MATH_EPS
        ) or (
            abs(dy - self._half_width) <= K_MATH_EPS
            and dx <= self._half_length + K_MATH_EPS
        )

    def distance_to_point(self, point):
        offset = Vec2d(point.x() - self._center.x(), point.y() - self._center.y())
        vec = offset.rotate_clockwise(self._heading)
        dx = abs(vec.x()) - self._half_length
        dy = abs(vec.y()) - self._half_width
        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return hypot(dx, dy)

    def get_position_in_box(self, point):
        offset = Vec2d(
            point.x() - self._center.x(),
            point.y() - self._center.y(),
        )
        vec = offset.rotate_clockwise(self._heading)
        x = vec.x()
        y = vec.y()
        box_x = self._half_length
        box_y = self._half_width
        if x >= box_x:
            gx = 1
        elif x <= -box_x:
            gx = -1
        else:
            gx = 0
        if y >= box_y:
            gy = 1
        elif y <= -box_y:
            gy = -1
        else:
            gy = 0
        return (x, y), (gx, gy)

    def distance_to_segment(self, segment):
        if segment.length() <= K_MATH_EPS:
            return self.distance_to_point(segment.start())
        box_x = self._half_length
        box_y = self._half_width

        print("segment length: ", segment.start().distance_to(segment.end()))

        # seg_start global frame to box frame
        (x1, y1), (gx1, gy1) = self.get_position_in_box(segment.start())
        if gx1 == 0 and gy1 == 0:
            return 0.0

        # seg_end global frame to box frame
        (x2, y2), (gx2, gy2) = self.get_position_in_box(segment.end())
        if gx2 == 0 and gy2 == 0:
            return 0.0

        print("segment length2: ", Vec2d(x1, y1).distance_to(Vec2d(x2, y2)))

        if gx1 < 0 or (gx1 == 0 and gx2 < 0):
            x1 = -x1
            gx1 = -gx1
            x2 = -x2
            gx2 = -gx2
        if gy1 < 0 or (gy1 == 0 and gy2 < 0):
            y1 = -y1
            gy1 = -gy1
            y2 = -y2
            gy2 = -gy2
        if gx1 < gy1 or (gx1 == gy1 and gx2 < gy2):
            x1, y1, gx1, gy1, x2, y2, gx2, gy2 = y1, x1, gy1, gx1, y2, x2, gy2, gx2
            box_x, box_y = box_y, box_x

        # not finish
        pass

    def distance_to_box(self, box):
        pass

    def distance_sqr_to_box(self, box):
        pass

    def has_overlap_to_segment(self, segment):
        pass

    def has_overlap_to_box(self, box):
        pass

    def get_aabox(self):
        dx1 = abs(self._cos_heading * self._half_length)
        dy1 = abs(self._sin_heading * self._half_length)
        dx2 = abs(self._sin_heading * self._half_width)
        dy2 = abs(self._cos_heading * self._half_width)
        return AABox2d(self._center, (dx1 + dx2) * 2.0, (dy1 + dy2) * 2.0)

    def rotate_from_center(self, rotate_angle):
        self._heading = normalize_angle(self._heading + rotate_angle)
        self._cos_heading = cos(self._heading)
        self._sin_heading = sin(self._heading)

    def shift(self, shift_vec):
        self._center += shift_vec

    def pt_seg_distance(self, query_x, query_y, start_x, start_y, end_x, end_y, length):
        x0 = query_x - start_x
        y0 = query_y - start_y
        dx = end_x - start_x
        dy = end_y - start_y
        proj = x0 * dx + y0 * dy
        if proj <= 0.0:
            return hypot(x0, y0)
        if proj >= length * length:
            return hypot(x0 - dx, y0 - dy)
        return abs(x0 * dy - y0 * dx) / length

    def pt_seg_distance2(self, query, start, end):
        seg = Segment2d(params=(start, end))
        return seg.distance_to(query)

    def pt_seg_distance3(self, query, start, end):
        seg = Segment2d(params=(start, end))
        if seg.length() <= K_MATH_EPS:
            return seg.start().distance_to(query)
        proj = seg.project_onto_unit(query)
        if proj <= 0.0:
            return seg.start().distance_to(query)
        if proj >= seg.length():
            return seg.end().distance_to(query)
        return abs(seg.product_onto_unit(query))

    def debug_string(self):
        return f"box2d ( center = {self._center.debug_string()}  length = {self._length}  width = {self._width}  heading={self._heading})"


def test1():
    pass


if __name__ == "__main__":
    test1()
