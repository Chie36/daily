import math

class Vec2d:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

    @staticmethod
    def create_unit_vec(angle):
        return Vec2d(math.cos(angle), math.sin(angle))

    def length(self):
        return math.sqrt(self.x_ * self.x_ + self.y_ * self.y_)

    def length_sqr(self):
        return self.x_ * self.x_ + self.y_ * self.y_

    def distance_to(self, other):
        return math.sqrt((self.x_ - other.x_) * (self.x_ - other.x_) +
                         (self.y_ - other.y_) * (self.y_ - other.y_))

    def distance_sqr_to(self, other):
        dx = self.x_ - other.x_
        dy = self.y_ - other.y_
        return dx * dx + dy * dy

    def normalize(self):
        l = self.length()
        if l > 1e-9:
            self.x_ /= l
            self.y_ /= l

    def __eq__(self, other):
        return (abs(self.x_ - other.x_) < 1e-9 and
                abs(self.y_ - other.y_) < 1e-9)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return Vec2d(self.x_ + other.x_, self.y_ + other.y_)

    def __sub__(self, other):
        return Vec2d(self.x_ - other.x_, self.y_ - other.y_)

    def __mul__(self, ratio):
        return Vec2d(self.x_ * ratio, self.y_ * ratio)

    def __truediv__(self, ratio):
        if abs(ratio) <= 1e-9:
            raise ValueError("ratio <= 1e-9")
        return Vec2d(self.x_ / ratio, self.y_ / ratio)

    def __iadd__(self, other):
        self.x_ += other.x_
        self.y_ += other.y_
        return self

    def __isub__(self, other):
        self.x_ -= other.x_
        self.y_ -= other.y_
        return self

    def __imul__(self, ratio):
        self.x_ *= ratio
        self.y_ *= ratio
        return self

    def __itruediv__(self, ratio):
        if abs(ratio) <= 1e-9:
            raise ValueError("ratio <= 1e-9")
        self.x_ /= ratio
        self.y_ /= ratio
        return self

    def cross_prod(self, other):
        return self.x_ * other.y_ - self.y_ * other.x_

    def inner_prod(self, other):
        return self.x_ * other.x_ + self.y_ * other.y_

    def rotate(self, angle):
        return Vec2d(self.x_ * math.cos(angle) - self.y_ * math.sin(angle),
                     self.x_ * math.sin(angle) + self.y_ * math.cos(angle))

    def self_rotate(self, angle):
        tmp_x = self.x_
        self.x_ = self.x_ * math.cos(angle) - self.y_ * math.sin(angle)
        self.y_ = tmp_x * math.sin(angle) + self.y_ * math.cos(angle)

    def __str__(self):
        return f"vec2d ( x = {self.x_:.6f}  y = {self.y_:.6f} )"