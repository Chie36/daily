import math

K_MATH_EPS = 1e-10


class Vec3d:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self._x = x
        self._y = y
        self._z = z

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def length(self):
        return math.sqrt(self.length_sqr())

    def length_sqr(self):
        return self._x * self._x + self._y * self._y + self._z * self._z

    def distance_to(self, other):
        return math.sqrt(self.distance_sqr_to(other))

    def distance_sqr_to(self, other):
        dx = self._x - other.x()
        dy = self._y - other.y()
        dz = self._z - other.z()
        return dx * dx + dy * dy + dz * dz

    def normalize(self):
        l = self.length()
        if l > K_MATH_EPS:
            self._x /= l
            self._y /= l
            self._z /= l

    def __eq__(self, other):
        return (
            abs(self._x - other.x()) < K_MATH_EPS
            and abs(self._y - other.y()) < K_MATH_EPS
            and abs(self._z - other.z()) < K_MATH_EPS
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Vec3d(self._x + other.x(), self._y + other.y(), self._z + other.z())

    def __sub__(self, other):
        return Vec3d(self._x - other.x(), self._y - other.y(), self._z - other.z())

    def __mul__(self, ratio):
        return Vec3d(self._x * ratio, self._y * ratio, self._z * ratio)

    def __truediv__(self, ratio):
        if abs(ratio) <= K_MATH_EPS:
            raise ValueError("ratio <= K_MATH_EPS")
        return Vec3d(self._x / ratio, self._y / ratio, self._z / ratio)

    def __iadd__(self, other):
        self._x += other.x()
        self._y += other.y()
        self._z += other.z()
        return self

    def __isub__(self, other):
        self._x -= other.x()
        self._y -= other.y()
        self._z -= other.z()
        return self

    def __imul__(self, ratio):
        self._x *= ratio
        self._y *= ratio
        self._z *= ratio
        return self

    def __itruediv__(self, ratio):
        if abs(ratio) <= K_MATH_EPS:
            raise ValueError("ratio <= K_MATH_EPS")
        self._x /= ratio
        self._y /= ratio
        self._z /= ratio
        return self

    def cross_prod(self, other):
        return Vec3d(
            self._y * other.z() - self._z * other.y(),
            self._z * other.x() - self._x * other.z(),
            self._x * other.y() - self._y * other.x(),
        )

    def inner_prod(self, other):
        return self._x * other.x() + self._y * other.y() + self._z * other.z()

    def rotate_xy(self, angle):
        new_x = self._x * math.cos(angle) - self._y * math.sin(angle)
        new_y = self._x * math.sin(angle) + self._y * math.cos(angle)
        return Vec3d(new_x, new_y, self._z)

    def rotate_xz(self, angle):
        new_x = self._x * math.cos(angle) - self._z * math.sin(angle)
        new_z = self._x * math.sin(angle) + self._z * math.cos(angle)
        return Vec3d(new_x, self._y, new_z)

    def rotate_yz(self, angle):
        new_y = self._y * math.cos(angle) - self._z * math.sin(angle)
        new_z = self._y * math.sin(angle) + self._z * math.cos(angle)
        return Vec3d(self._x, new_y, new_z)

    def __str__(self):
        return f"vec3d ( x = {self._x:.3f}  y = {self._y:.3f}  z = {self._z:.3f} )"


def test():
    # Create Vec3d instances
    v1 = Vec3d(3, 4, 5)
    v2 = Vec3d(1, 2, 3)

    # 1. Print vector coordinates
    print(f"v1: {v1}, v2: {v2}")

    # 2. Length of vectors
    print(f"v1 length: {v1.length():.3f}, v2 length: {v2.length():.3f}")

    # 3. Length squared
    print(
        f"v1 length squared: {v1.length_sqr():.3f}, v2 length squared: {v2.length_sqr():.3f}"
    )

    # 4. Distance between v1 and v2
    print(f"Distance between v1 and v2: {v1.distance_to(v2):.3f}")

    # 5. Distance squared
    print(f"Distance squared: {v1.distance_sqr_to(v2):.3f}")

    # 6. Normalize v1
    v1.normalize()
    print(f"Normalized v1: {v1}")

    # 7. Vector addition
    v3 = v1 + v2
    print(f"v1 + v2: {v3}")

    # 8. Vector subtraction
    v4 = v1 - v2
    print(f"v1 - v2: {v4}")

    # 9. Scalar multiplication
    v5 = v1 * 3
    print(f"v1 * 3: {v5}")

    # 10. Scalar division
    v6 = v2 / 2
    print(f"v2 / 2: {v6}")

    # 11. Dot product
    print(f"v1 · v2: {v1.inner_prod(v2):.3f}")

    # 12. Cross product
    cross_prod = v1.cross_prod(v2)
    print(f"v1 × v2: ({cross_prod.x}, {cross_prod.y}, {cross_prod.z})")

    # 13. Rotate v1 in the XY plane by 45 degrees
    v7 = v1.rotate_xy(math.pi / 4)
    print(f"v1 rotated 45° in XY plane: {v7}")

    # 14. Rotate v1 in the XZ plane by 90 degrees
    v8 = v1.rotate_xz(math.pi / 2)
    print(f"v1 rotated 90° in XZ plane: {v8}")

    # 15. Rotate v1 in the YZ plane by 90 degrees
    v9 = v1.rotate_yz(math.pi / 2)
    print(f"v1 rotated 90° in YZ plane: {v9}")

    # 16. Vector equality
    v10 = Vec3d(0.6, 0.8, 1.0)
    print(f"v1 == v10: {v1 == v10}")

    # 17. Vector inequality
    print(f"v1 != v10: {v1 != v10}")


if __name__ == "__main__":
    test()
