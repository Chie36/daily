import numpy as np
import matplotlib.pyplot as plt


class BsplineFit:
    def __init__(self, points, order, interval):
        self.control_points = points
        self.p = order
        self.n = points.shape[0] - 1
        self.m = self.n + self.p + 1
        self.interval = interval
        self.u = self._generate_knot_vector()

    def _generate_knot_vector(self):
        u = np.zeros(self.m + 1)
        for i in range(self.m + 1):
            if i <= self.p:
                u[i] = -self.p + i * self.interval
            else:
                u[i] = u[i - 1] + self.interval
        return u

    def set_knot(self, knot):
        self.u = knot

    def get_knot(self):
        return self.u

    def get_control_point(self):
        return self.control_points

    def get_interval(self):
        return self.interval

    def get_time_span(self):
        return (self.u[self.p], self.u[self.m - self.p])

    def get_head_tail_pts(self):
        head = self.evaluate_de_boor(self.u[self.p])
        tail = self.evaluate_de_boor(self.u[self.m - self.p])
        return head, tail

    def get_derivative_control_points(self):
        ctrl_pts = np.zeros(
            (self.control_points.shape[0] - 1, self.control_points.shape[1])
        )
        for i in range(ctrl_pts.shape[0]):
            ctrl_pts[i] = (
                self.p
                * (self.control_points[i + 1] - self.control_points[i])
                / (self.u[i + self.p + 1] - self.u[i + 1])
            )
        return ctrl_pts

    def evaluate_de_boor(self, u):
        ub = np.clip(u, self.u[self.p], self.u[self.m - self.p])
        k = self.p
        while self.u[k + 1] < ub:
            k += 1
        d = [self.control_points[k - self.p + i] for i in range(self.p + 1)]
        for r in range(1, self.p + 1):
            for i in range(self.p, r - 1, -1):
                alpha = (ub - self.u[i + k - self.p]) / (
                    self.u[i + 1 + k - r] - self.u[i + k - self.p]
                )
                d[i] = (1 - alpha) * d[i - 1] + alpha * d[i]
        return d[self.p]

    def evaluate_de_boor_t(self, t):
        return self.evaluate_de_boor(t + self.u[self.p])

    def get_derivative(self):
        ctp = self.get_derivative_control_points()
        derivative = BsplineFit(ctp, self.p - 1, self.interval)
        derivative.set_knot(self.u[1:-1])
        return derivative

    def parameterize_to_bspline(self, ts, point_set, start_end_derivative):
        K = len(point_set)
        dims = point_set[0].shape[0]

        condition_nums = K + start_end_derivative.shape[0]
        ctrl_pts_nums = self.p + K - 1

        A = np.zeros((condition_nums, ctrl_pts_nums))

        if self.p == 2:
            prow = np.array([1, 1]) * (1 / 2.0) * 1
            vrow = np.array([-2, 2]) * (1 / 2.0) * (1 / ts)
            for i in range(K):
                A[i, i : i + self.p] = prow
            A[K, : self.p] = vrow
            A[K + 1, K - 1 : ctrl_pts_nums] = vrow

        if self.p == 3:
            prow = np.array([1, 4, 1]) * (1 / 6.0) * 1
            vrow = np.array([-3, 0, 3]) * (1 / 6.0) * (1 / ts)
            arow = np.array([3, -6, 3]) * (1 / 6.0) * (2 / ts / ts)
            for i in range(K):
                A[i, i : i + self.p] = prow
            A[K, : self.p] = vrow
            A[K + 1, K - 1 : ctrl_pts_nums] = vrow
            A[K + 2, : self.p] = arow
            A[K + 3, K - 1 : ctrl_pts_nums] = arow

        elif self.p == 4:
            prow = np.array([1, 11, 11, 1]) * (1 / 24.0) * 1
            vrow = np.array([-4, -12, 12, 4]) * (1 / 24.0) * (1 / ts)
            arow = np.array([6, -6, -6, 6]) * (1 / 24.0) * (2 / ts / ts)
            jrow = np.array([-4, 12, -12, 4]) * (1 / 24.0) * (6 / ts / ts / ts)
            for i in range(K):
                A[i, i : i + self.p] = prow
            A[K, : self.p] = vrow
            A[K + 1, K - 1 : ctrl_pts_nums] = vrow
            A[K + 2, : self.p] = arow
            A[K + 3, K - 1 : ctrl_pts_nums] = arow
            A[K + 4, : self.p] = jrow
            A[K + 5, K - 1 : ctrl_pts_nums] = jrow

        b_all = np.zeros((condition_nums, dims))
        b_all[:K, :] = point_set[:K, :]  # positions
        b_all[K:condition_nums, :] = start_end_derivative  # derivatives

        ctrl_pts = np.linalg.lstsq(A, b_all, rcond=None)[0]
        return ctrl_pts

    def generate_path(self, dt):
        duration = self.u[self.m - self.p] - self.u[self.p]
        return [
            self.evaluate_de_boor_t(tc) for tc in np.arange(0.0, duration + 1e-3, dt)
        ]


def test1():
    control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 0], [5, 2]])
    spline = BsplineFit(control_points, 3, 1.0)

    control_points = spline.get_control_point()
    plt.plot(control_points[:, 0], control_points[:, 1], "ro-", label="Control Points")

    path = spline.generate_path(0.1)
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], "b-", label="B-Spline Path")

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Non-Uniform B-Spline Curve")
    plt.grid(True)
    plt.show()


def test2():
    point_set = np.array(
        [
            [229158.99468523, 3683297.26641572],
            [229159.20622795, 3683296.92693129],
            [229159.45753626, 3683296.61573339],
            [229159.74486972, 3683296.33745388],
            [229160.06395166, 3683296.09623466],
            [229160.41003289, 3683295.89566601],
            [229160.77796235, 3683295.73873320],
            [229161.16226381, 3683295.62777199],
            [229161.54817290, 3683295.52253808],
            [229161.93560427, 3683295.42305475],
            [229162.32447224, 3683295.32934399],
            [229162.71469079, 3683295.24142653],
        ]
    )
    start_end_derivative = np.array(
        [
            [0.52885678, -0.84871108],
            [0.97554639, -0.21979365],
            [0.12426750, 0.08839542],
            [0.00422060, 0.01810406],
        ]
    )

    ts = 0.4
    order = 3

    control_points = np.array([[]])
    spline = BsplineFit(control_points, order, ts)

    control_points = spline.parameterize_to_bspline(ts, point_set, start_end_derivative)
    print("control_points:\n", control_points)

    spline_with_control_points = BsplineFit(control_points, order, ts)
    path = spline_with_control_points.generate_path(0.05)

    point_set = np.array(point_set)
    path = np.array(path)

    plt.figure(figsize=(8, 6))
    plt.plot(point_set[:, 0], point_set[:, 1], "go", label="pts", markersize=8)
    plt.plot(
        control_points[:, 0],
        control_points[:, 1],
        "r*",
        label="ctrl pts",
        markersize=8,
    )
    plt.plot(path[:, 0], path[:, 1], "b-", label="bspline")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("B-Spline test")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # test1()
    test2()
