import numpy as np
import matplotlib.pyplot as plt
import nlopt

from bspline_fit import BsplineFitter


class Config:
    def __init__(self):
        self.w_smoothness = 20.0
        self.w_distance = 2.0
        self.w_guide = 10.0
        self.w_endpoint = 10.0
        self.safe_dist = 3.0


class OccMap:

    def __init__(self):
        pass

    def get_point_distance(self, pt):
        pass

    def get_point_gradient_direction(self, pt):
        pass


class BsplineOptimizer:
    def __init__(self, ctrl_pts, om, conf, is_end_fix):
        self.is_log_debug = False

        self.om = om
        self.conf = conf
        self.ctrl_pts = np.array(ctrl_pts)
        self.pt_dim = self.ctrl_pts.shape[1]
        self.pt_num = self.ctrl_pts.shape[0]
        self.order = 3  # tmp
        self.is_end_fix = is_end_fix
        self.variable_num = (
            self.pt_dim * (self.pt_num - self.order)
            if self.is_end_fix
            else max(0, self.pt_dim * (self.pt_num - 2 * self.order))
        )

        print(
            f"bspline info: pt_dim({self.pt_dim}), pt_num({self.pt_num}), order({self.order}), is_end_fix({self.is_end_fix}), variable_num({self.variable_num})"
        )

        self.guide_pts = self.ctrl_pts.copy()
        self.set_opt()

    def set_guide(self, guide_pts):
        self.guide_pts = guide_pts.copy()

    def set_opt(self):
        # iter
        self.iter_num = 0
        self.min_cost = float("inf")
        self.ctrl_pts_in_iter = np.zeros((self.pt_num, self.pt_dim))

        # solution
        self.end_idx = self.pt_num if self.is_end_fix else self.pt_num - self.order
        self.init_solution = self.ctrl_pts[self.order : self.end_idx].flatten().copy()
        self.final_solution = np.zeros(self.variable_num)
        self.best_solution = np.zeros(self.variable_num)
        self.lb = self.init_solution - 1e5
        self.ub = self.init_solution + 1e5

        # opt
        self.opt = nlopt.opt(nlopt.LD_LBFGS, self.variable_num)
        self.opt.set_min_objective(self.cost_function)
        self.opt.set_maxeval(50)
        self.opt.set_xtol_rel(1e-5)
        self.opt.set_lower_bounds(self.lb)
        self.opt.set_upper_bounds(self.ub)

        print(f"init_solution({self.variable_num}):\n{self.init_solution}")

    def optimize(self):
        try:
            self.final_solution = self.opt.optimize(self.init_solution)
            print(
                f"min_cost({self.opt.last_optimum_value():.3f}), code({self.opt.last_optimize_result()}), final_solution({self.variable_num}):\n{self.final_solution}"
            )
            for i in range(self.order, self.end_idx):
                offset = self.pt_dim * (i - self.order)
                self.ctrl_pts[i, :] = np.copy(
                    self.best_solution[offset : offset + self.pt_dim]
                )
        except Exception as e:
            print(f"NLopt exception: {str(e)}")

        return self.ctrl_pts

    def cost_function(self, x, grad):
        self.iter_num += 1
        cost = np.array([0.0])
        self.combine_cost(x, grad, cost)
        cost_val = cost[0]
        if cost_val < self.min_cost:
            self.min_cost = cost_val
            self.best_solution = x.copy()
        return cost_val

    def combine_cost(self, x, grad, cost):
        feas_pt_nums = self.variable_num // self.pt_dim
        if not self.is_end_fix:
            self.ctrl_pts_in_iter[: self.order, :] = np.copy(
                self.ctrl_pts[: self.order, :]
            )
            self.ctrl_pts_in_iter[self.order + feas_pt_nums :, :] = np.copy(
                self.ctrl_pts[self.pt_num - self.order :, :]
            )
        else:
            for i in range(self.order):
                for j in range(self.pt_dim):
                    self.ctrl_pts_in_iter[i, j] = self.ctrl_pts[i, j]

        x_reshaped = np.reshape(x, (feas_pt_nums, self.pt_dim))
        self.ctrl_pts_in_iter[self.order : self.order + feas_pt_nums, :] = np.copy(
            x_reshaped
        )

        log = ""

        def update(cost, grad, weight, f, g, sign):
            nonlocal log
            cost[0] += weight * f[0]
            grad[: feas_pt_nums * self.pt_dim] += (
                weight * g[self.order : self.order + feas_pt_nums, :].flatten().copy()
            )
            log += f" {sign}:{weight:.1f} x {f[0]:.3f}"

        def calc_cost(func, sign, weight):
            nonlocal cost
            nonlocal grad
            cost_i = np.array([0.0])
            grad_i = np.zeros((self.pt_num, self.pt_dim))
            self.__getattribute__(func)(self.ctrl_pts_in_iter.copy(), cost_i, grad_i)
            update(cost, grad, weight, cost_i, grad_i, sign)

        calc_cost("calc_smoothness_cost", "SMOOTHNESS", self.conf.w_smoothness)
        # calc_cost("calc_distance_cost", "DISTANCE", self.conf.w_distance)
        calc_cost("calc_guide_cost", "GUIDE", self.conf.w_guide)
        calc_cost("calc_endpoint_cost", "ENDPOINT", self.conf.w_endpoint)

        print(
            f"iter({self.iter_num}) var_num:{self.variable_num}, cost:{cost[0]:.3f}"
            + log
        )

    def calc_smoothness_cost(self, q, cost, gradient):
        s_str = f"\nCalcSmoothnessCost:"
        for i in range(self.pt_num - self.order):
            jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]
            cur_cost = jerk.dot(jerk)
            s_str += f"\n\tcp:{i}-{i + 3}, cost:{cur_cost:.3f}, jerk:({jerk[0]:.3f},{jerk[1]:.3f})"
            cost += cur_cost
            temp_j = 2.0 * jerk
            gradient[i + 0] += -temp_j
            gradient[i + 1] += 3.0 * temp_j
            gradient[i + 2] += -3.0 * temp_j
            gradient[i + 3] += temp_j
        if self.is_log_debug:
            print(s_str)

    def calc_distance_cost(self, q, cost, gradient):
        d_str = f"\nCalcDistanceCost safe_dist:{self.conf.safe_dist:.3f}"
        for i in range(self.order, self.end_idx):
            dist = self.om.get_point_distance((q[i][0], q[i][1]))
            dist_grad_ad2 = self.om.get_point_gradient_direction((q[i][0], q[i][1]))
            dist_grad = np.array(dist_grad_ad2)
            dist_grad /= np.linalg.norm(dist_grad)
            d_str += f"\n\tcp:{i}, dist:{dist:.3f}, pose:({q[i][0]:.3f}, {q[i][1]:.3f}), dist_grad:({dist_grad_ad2[0]}, {dist_grad_ad2[1]})"
            if dist < self.conf.safe_dist:
                cost += (dist - self.conf.safe_dist) ** 2
                gradient[i] += 2.0 * (dist - self.conf.safe_dist) * dist_grad
        if self.is_log_debug:
            print(d_str)

    def calc_guide_cost(self, q, cost, gradient):
        g_str = f"\nCalcGuideCost:"
        for i in range(self.order, self.pt_num - self.order):
            delta = q[i] - self.guide_pts[i]
            cur_cost = delta.dot(delta)
            g_str += f"\n\tcp:{i}, delta:({delta[0]:.3f}, {delta[1]:.3f}), dis:{cur_cost:.3f}"
            cost += cur_cost
            gradient[i] += 2 * delta
        if self.is_log_debug:
            print(g_str)

    def calc_endpoint_cost(self, q, cost, gradient):
        if not self.is_end_fix:
            return
        e_str = f"\nCalcEndpointCost:"
        cp_3 = self.ctrl_pts[self.pt_num - 3]
        cp_2 = self.ctrl_pts[self.pt_num - 2]
        cp_1 = self.ctrl_pts[self.pt_num - 1]
        ref = 1 / 6.0 * (cp_3 + 4 * cp_3 + cp_1)
        q_3 = q[self.pt_num - 3]
        q_2 = q[self.pt_num - 2]
        q_1 = q[self.pt_num - 1]
        opt = 1 / 6.0 * (q_3 + 4 * q_2 + q_1)
        dq = opt - ref
        cur_cost = dq.dot(dq)
        e_str += f"\n\tdq:({dq[0]:.3f}, {dq[1]:.3f}), dis:{cur_cost:.3f}"
        cost += cur_cost
        gradient[self.pt_num - 3] += 2 * dq * (1 / 6.0)
        gradient[self.pt_num - 2] += 2 * dq * (4 / 6.0)
        gradient[self.pt_num - 1] += 2 * dq * (1 / 6.0)
        if self.is_log_debug:
            print(e_str)


def test():
    point_set_str = """ 
     229158.58474422  3683297.48702254 
     229158.80728339  3683297.15464276 
     229159.06864118  3683296.85183571 
     229159.36492754  3683296.58310835 
     229159.69173257  3683296.35246041 
     229160.04419212  3683296.16332484 
     229160.41706021  3683296.01851672 
     229160.80478708  3683295.92019137 
     229161.20160182  3683295.86981225 
     229161.60159828  3683295.86812921 
     229161.99882293  3683295.91516730 
    """
    point_set = np.array(
        [list(map(float, line.split())) for line in point_set_str.strip().splitlines()]
    )
    start_end_derivative_str = """ 
     0.55634793  -0.83094945 
     0.99306161  0.11759521 
     0.12130817  0.09241478 
     -0.00866192  0.15225352 
    """
    start_end_derivative = np.array(
        [
            list(map(float, line.split()))
            for line in start_end_derivative_str.strip().splitlines()
        ]
    )

    interval = 0.4
    order = 3

    control_points = np.array([[]])
    fitter = BsplineFitter(control_points, order, interval)
    control_points_fit = fitter.parameterize_to_bspline(
        point_set, start_end_derivative, order, interval
    )

    optimizer = BsplineOptimizer(
        control_points_fit,
        OccMap(),
        Config(),
        False,
    )
    control_points_opt = optimizer.optimize()

    plt.figure(figsize=(8, 6))
    plt.plot(
        control_points_fit[:, 0],
        control_points_fit[:, 1],
        "r*",
        label="Control Points Fit",
        markersize=8,
    )
    plt.plot(
        control_points_opt[:, 0],
        control_points_opt[:, 1],
        "b*",
        label="Control Points Opt",
        markersize=8,
    )

    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("test")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    test()
