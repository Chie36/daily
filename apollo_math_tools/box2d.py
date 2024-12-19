import math
from vec2d import Vec2d
from segment2d import Segment2d
from aabox2d import AABox2d
from util import cross_prod, normalize_angle

import matplotlib.pyplot as plt

K_MATH_EPS = 1e-10


class Box2d:
    def __init__(self, center, heading, length, width):
        self._center = center
        self._length = length
        self._width = width
        self._half_length = length / 2.0
        self._half_width = width / 2.0
        self._heading = heading
        self._cos_heading = math.cos(heading)
        self._sin_heading = math.sin(heading)
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
        return math.hypot(self._length, self._width)

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
        print("get_all_corners", [c.debug_string() for c in self._corners])
        return self._corners

    def get_all_corners_list(self):
        x_list = []
        y_list = []
        for corner in self._corners:
            x_list.append(corner.x())
            y_list.append(corner.y())
        x_list.append(self._corners[0].x())
        y_list.append(self._corners[0].y())
        return x_list, y_list

    def is_point_in(self, point):
        offset = point - self._center
        vec = offset.rotate_clockwise(self._heading)
        return (
            abs(vec.x()) <= self._half_length + K_MATH_EPS
            and abs(vec.y()) <= self._half_width + K_MATH_EPS
        )

    def is_point_on_boundary(self, point):
        offset = point - self._center
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
        offset = point - self._center
        vec = offset.rotate_clockwise(self._heading)
        dx = abs(vec.x()) - self._half_length
        dy = abs(vec.y()) - self._half_width
        if dx <= 0.0:
            return max(0.0, dy)
        if dy <= 0.0:
            return dx
        return math.hypot(dx, dy)

    def get_position_in_box(self, point):
        offset = point - self._center
        vec = offset.rotate_clockwise(self._heading)
        x = vec.x()
        y = vec.y()
        gx = 1 if x >= self._half_length else -1 if x <= -self._half_length else 0
        gy = 1 if y >= self._half_width else -1 if y <= -self._half_width else 0
        return (x, y), (gx, gy)

    def distance_to_segment(self, segment):
        if segment.length() <= K_MATH_EPS:
            return self.distance_to_point(segment.start())
        box_x = self._half_length
        box_y = self._half_width

        (x1, y1), (gx1, gy1) = self.get_position_in_box(segment.start())
        if gx1 == 0 and gy1 == 0:
            return 0.0

        (x2, y2), (gx2, gy2) = self.get_position_in_box(segment.end())
        if gx2 == 0 and gy2 == 0:
            return 0.0

        if gx1 < 0 or (gx1 == 0 and gx2 < 0):
            x1, gx1, x2, gx2 = -x1, -gx1, -x2, -gx2
        if gy1 < 0 or (gy1 == 0 and gy2 < 0):
            y1, gy1, y2, gy2 = -y1, -gy1, -y2, -gy2
        if gx1 < gy1 or (gx1 == gy1 and gx2 < gy2):
            x1, y1, gx1, gy1, x2, y2, gx2, gy2 = y1, x1, gy1, gx1, y2, x2, gy2, gx2
            box_x, box_y = box_y, box_x

        rt_seg_dis = self.pt_seg_distance(
            Vec2d(box_x, box_y), Vec2d(x1, y1), Vec2d(x2, y2)
        )
        rb_seg_dis = self.pt_seg_distance(
            Vec2d(box_x, -box_y), Vec2d(x1, y1), Vec2d(x2, y2)
        )
        lt_seg_dis = self.pt_seg_distance(
            Vec2d(-box_x, box_y), Vec2d(x1, y1), Vec2d(x2, y2)
        )
        rt_seg_cross_prod = cross_prod(
            Vec2d(x1, y1), Vec2d(x2, y2), Vec2d(box_x, box_y)
        )
        rd_seg_cross_prod = cross_prod(
            Vec2d(x1, y1), Vec2d(x2, y2), Vec2d(box_x, -box_y)
        )
        if gx1 == 1 and gy1 == 1:
            case_value = gx2 * 3 + gy2
            if case_value == 4:
                return rt_seg_dis
            elif case_value == 3:
                return (x2 - box_x) if x1 > x2 else rt_seg_dis
            elif case_value == 2:
                return rb_seg_dis if x1 > x2 else rt_seg_dis
            elif case_value == -1:
                return 0.0 if rd_seg_cross_prod >= 0.0 else rb_seg_dis
            elif case_value == -4:
                return (
                    rb_seg_dis
                    if rd_seg_cross_prod <= 0.0
                    else (0.0 if rd_seg_cross_prod <= 0.0 else lt_seg_dis)
                )
        else:
            case_value = gx2 * 3 + gy2
            if case_value == 4:
                return (x1 - box_x) if x1 < x2 else rt_seg_dis
            elif case_value == 3:
                return min(x1, x2) - box_x
            elif case_value == 1 or case_value == -2:
                return 0.0 if rt_seg_cross_prod <= 0.0 else rt_seg_dis
            elif case_value == -3:
                return 0.0
        print(f"unimplemented state: gx1: {gx1}, gy1: {gy1}, gx2: {gx2}, gy2: {gy2}")
        return 0.0

    def distance_to_box(self, box):
        # polygon after
        pass

    def distance_sqr_to_box(self, box):
        # polygon after
        pass

    def has_overlap_to_segment(self, segment):
        if segment.length() <= K_MATH_EPS:
            return self.is_point_in(segment.start())
        return self.distance_to_segment(segment) <= K_MATH_EPS

    def has_overlap_to_box(self, box):
        shift_x = box.center_x() - self._center.x()
        shift_y = box.center_y() - self._center.y()

        # 自box长轴向量(dx1, dy1)
        dx1 = self._cos_heading * self._half_length
        dy1 = self._sin_heading * self._half_length
        # 自box短轴向量(dx2, dy2)
        dx2 = self._sin_heading * self._half_width
        dy2 = -self._cos_heading * self._half_width

        # 他box长轴向量(dx3, dy3)
        dx3 = box.cos_heading() * box.half_length()
        dy3 = box.sin_heading() * box.half_length()
        # 他box短轴向量(dx4, dy4)
        dx4 = box.sin_heading() * box.half_width()
        dy4 = -box.cos_heading() * box.half_width()

        # (a, b)是他box中心位置在自box坐标系下的坐标
        a = shift_x * self._cos_heading + shift_y * self._sin_heading
        b = shift_x * self._sin_heading - shift_y * self._cos_heading
        # (c, d)是自box中心位置在他box坐标系下的坐标
        c = shift_x * box.cos_heading() + shift_y * box.sin_heading()
        d = shift_x * box.sin_heading() - shift_y * box.cos_heading()

        # q是他box长轴向量(dx3, dy3)在自box坐标系中长轴方向的投影
        q = dx3 * self._cos_heading + dy3 * self._sin_heading
        # w是他box长轴向量(dx3, dy3)在自box坐标系中短轴方向的投影
        w = dx3 * self._sin_heading - dy3 * self._cos_heading

        # e是他box短轴向量(dx4, dy4)在自box坐标系中长轴方向的投影
        e = dx4 * self._cos_heading + dy4 * self._sin_heading
        # r是他box短轴向量(dx4, dy4)在自box坐标系中短轴方向的投影
        r = dx4 * self._sin_heading - dy4 * self._cos_heading

        # t是自box长轴向量(dx1, dy1)在他box坐标系中长轴方向的投影
        t = dx1 * box.cos_heading() + dy1 * box.sin_heading()
        # u是自box长轴向量(dx1, dy1)在他box坐标系中短轴方向的投影
        u = dx2 * box.cos_heading() + dy2 * box.sin_heading()

        # i是自box短轴向量(dx2, dy2)在他box坐标系中长轴方向的投影
        i = dx1 * box.sin_heading() - dy1 * box.cos_heading()
        # o是自box短轴向量(dx2, dy2)在他box坐标系中短轴方向的投影
        o = dx2 * box.sin_heading() - dy2 * box.cos_heading()

        # SAT算法
        return (
            abs(a) <= abs(q) + abs(e) + self._half_length
            and abs(b) <= abs(w) + abs(r) + self._half_width
            and abs(c) <= abs(t) + abs(u) + box.half_length()
            and abs(d) <= abs(i) + abs(o) + box.half_width()
        )

    def get_aabox(self):
        dx1 = abs(self._cos_heading * self._half_length)
        dy1 = abs(self._sin_heading * self._half_length)
        dx2 = abs(self._sin_heading * self._half_width)
        dy2 = abs(self._cos_heading * self._half_width)
        return AABox2d(self._center, (dx1 + dx2) * 2.0, (dy1 + dy2) * 2.0)

    def rotate_from_center(self, rotate_angle):
        self._heading = normalize_angle(self._heading + rotate_angle)
        self._cos_heading = math.cos(self._heading)
        self._sin_heading = math.sin(self._heading)

    def shift(self, shift_vec):
        self._center += shift_vec

    def pt_seg_distance(self, query, start, end):
        seg = Segment2d(params=(start, end))
        return seg.distance_to(query)

    def debug_string(self):
        return f"box2d ( center = {self._center.debug_string()}  length = {self._length}  width = {self._width}  heading={self._heading})"


def test1():
    segment_start = Vec2d(2.5, 1.5)
    segment_end = Vec2d(-1.0, 1.0)
    segment = Segment2d((segment_start, segment_end))

    bounding_box = Box2d(Vec2d(0, 0), 0, 1, 1)
    distance = bounding_box.distance_to_segment(segment)
    print("Distance to segment:", distance)

    plt.figure()
    plt.plot(
        [segment_start.x(), segment_end.x()], [segment_start.y(), segment_end.y()], "b-"
    )
    x_list, y_list = bounding_box.get_all_corners_list()
    plt.plot(x_list, y_list, "g-")
    plt.grid()
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Point to Segment Distance")
    plt.show()


def test2():
    box1 = Box2d(Vec2d(0, 0), 0, 5, 5)
    box2 = Box2d(Vec2d(5, 5), 0, 5, 5)

    overlap_result = box1.has_overlap_to_box(box2)
    print("Has overlap:", overlap_result)

    plt.figure()
    x1_list, y1_list = box1.get_all_corners_list()
    plt.plot(x1_list, y1_list, "g-")
    x2_list, y2_list = box2.get_all_corners_list()
    plt.plot(x2_list, y2_list, "b-")
    plt.grid()
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Point to Segment Distance")
    plt.show()


if __name__ == "__main__":
    # test1()
    test2()
