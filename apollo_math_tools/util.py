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


def wrap_angle(angle):
    new_angle = angle % (2 * math.pi)
    return new_angle + 2 * math.pi if new_angle < 0 else new_angle


def normalize_angle(angle):
    new_angle = angle % (2 * math.pi)
    return (new_angle + 2 * math.pi if new_angle < 0 else new_angle) - math.pi


def angle_diff(from_angle, to_angle):
    return normalize_angle(to_angle - from_angle)


def test():
    # 示例1：计算两个向量的叉积
    point1 = Vec2d(1, 2)
    point2 = Vec2d(3, 4)
    point3 = Vec2d(5, 6)
    cross_product_result = cross_prod(point1, point2, point3)
    print("Cross product result:", cross_product_result)

    # 示例2：计算两个向量的点积
    point1 = Vec2d(1, 2)
    point2 = Vec2d(3, 4)
    point3 = Vec2d(5, 6)
    inner_product_result = inner_prod(point1, point2, point3)
    print("Inner product result:", inner_product_result)

    # 示例3：判断一个值是否在两个边界之间（考虑误差范围）
    val = 2.5
    bound1 = 2.0
    bound2 = 3.0
    is_within_result = is_within(val, bound1, bound2)
    print("Is within bounds:", is_within_result)


if __name__ == "__main__":
    test()
