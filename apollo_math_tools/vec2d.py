import math

K_MATH_EPS = 1e-10


class Vec2d:
    def __init__(self, x=0.0, y=0.0):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    @staticmethod
    def create_unit_vec(angle):
        return Vec2d(math.cos(angle), math.sin(angle))

    def angle(self):
        return math.atan2(self._y, self._x)

    def length(self):
        return math.sqrt(self.length_sqr())

    def length_sqr(self):
        return self._x * self._x + self._y * self._y

    def distance_to(self, other):
        return math.sqrt(self.distance_sqr_to(other))

    def distance_sqr_to(self, other):
        dx = self._x - other.x()
        dy = self._y - other.y()
        return dx * dx + dy * dy

    def normalize(self):
        l = self.length()
        if l > K_MATH_EPS:
            self._x /= l
            self._y /= l

    def __eq__(self, other):
        return (
            abs(self._x - other.x()) < K_MATH_EPS
            and abs(self._y - other.y()) < K_MATH_EPS
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Vec2d(self._x + other.x(), self._y + other.y())

    def __sub__(self, other):
        return Vec2d(self._x - other.x(), self._y - other.y())

    def __mul__(self, ratio):
        return Vec2d(self._x * ratio, self._y * ratio)

    def __truediv__(self, ratio):
        if abs(ratio) <= K_MATH_EPS:
            raise ValueError("ratio <= K_MATH_EPS")
        return Vec2d(self._x / ratio, self._y / ratio)

    def __iadd__(self, other):
        self._x += other.x()
        self._y += other.y()
        return self

    def __isub__(self, other):
        self._x -= other.x()
        self._y -= other.y()
        return self

    def __imul__(self, ratio):
        self._x *= ratio
        self._y *= ratio
        return self

    def __itruediv__(self, ratio):
        if abs(ratio) <= K_MATH_EPS:
            raise ValueError("ratio <= K_MATH_EPS")
        self._x /= ratio
        self._y /= ratio
        return self

    def cross_prod(self, other):
        return self._x * other.y() - self._y * other.x()

    def inner_prod(self, other):
        return self._x * other.x() + self._y * other.y()

    def rotate(self, angle):
        return Vec2d(
            self._x * math.cos(angle) - self._y * math.sin(angle),
            self._x * math.sin(angle) + self._y * math.cos(angle),
        )

    def self_rotate(self, angle):
        tmp_x = self._x
        self._x = self._x * math.cos(angle) - self._y * math.sin(angle)
        self._y = tmp_x * math.sin(angle) + self._y * math.cos(angle)

    def __str__(self):
        return f"vec2d ( x = {self._x:.3f}  y = {self._y:.3f} )"


def test():
    # Create Vec2d instances
    v1 = Vec2d(3, 4)
    v2 = Vec2d(1, 2)

    # 1. Print vector coordinates
    print(f"v1: {v1}, v2: {v2}")

    # 2. Length of vectors
    print(f"v1 length: {v1.length():.3f}, v2 length: {v2.length():.3f}")

    # 3. Length squared
    print(
        f"v1 length squared: {v1.length_sqr():.3f}, v2 length squared: {v2.length_sqr():.3f}"
    )

    # 4. Angle in radians
    print(f"v1 angle: {v1.angle():.3f}, v2 angle: {v2.angle():.3f}")

    # 5. Distance between v1 and v2
    print(f"Distance between v1 and v2: {v1.distance_to(v2):.3f}")

    # 6. Distance squared
    print(f"Distance squared: {v1.distance_sqr_to(v2):.3f}")

    # 7. Normalize v1
    v1.normalize()
    print(f"Normalized v1: {v1}")

    # 8. Vector addition
    v3 = v1 + v2
    print(f"v1 + v2: {v3}")

    # 9. Vector subtraction
    v4 = v1 - v2
    print(f"v1 - v2: {v4}")

    # 10. Scalar multiplication
    v5 = v1 * 3
    print(f"v1 * 3: {v5}")

    # 11. Scalar division
    v6 = v2 / 2
    print(f"v2 / 2: {v6}")

    # 12. Dot product
    print(f"v1 · v2: {v1.inner_prod(v2):.3f}")

    # 13. Cross product
    print(f"v1 × v2: {v1.cross_prod(v2):.3f}")

    # 14. Rotate v1 by 45 degrees
    v7 = v1.rotate(math.pi / 4)
    print(f"v1 rotated 45°: {v7}")

    # 15. Self-rotate v1 by 90 degrees
    v1.self_rotate(math.pi / 2)
    print(f"v1 after self-rotation 90°: {v1}")

    # 16. Create unit vector at 45 degrees
    unit_vec = Vec2d.create_unit_vec(math.pi / 4)
    print(f"Unit vector at 45°: {unit_vec}")

    # 17. Vector equality
    v8 = Vec2d(0.6, 0.8)
    print(f"v1 == v8: {v1 == v8}")

    # 18. Vector inequality
    print(f"v1 != v8: {v1 != v8}")


if __name__ == "__main__":
    test()
