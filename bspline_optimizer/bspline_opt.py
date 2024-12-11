import numpy as np
import nlopt

class BsplineOptimizer:
    def __init__(self, ctrl_pts, om, ts, end_fix):
        self.om_ = om
        self.control_points_ = np.array(ctrl_pts)
        self.pt_dim_ = self.control_points_.shape[1]
        self.pt_num_ = self.control_points_.shape[0]
        self.order_ = 3
        self.interval_ = ts
        self.guide_pts_ = np.zeros((self.pt_num_, self.pt_dim_))
        for i in range(self.pt_num_):
            for j in range(self.pt_dim_):
                self.guide_pts_[i][j] = self.control_points_[i][j]
        self.end_fix_ = end_fix
        print(f"bspline info pt_dim:{self.pt_dim_}, pt_num:{self.pt_num_}, order:{self.order_}, end_fix:{end_fix}")

    def set_config(self, conf):
        self.conf_ = conf

    def bspline_optimize_traj(self):
        if self.log_debug:
            pts_str = "before control_points:"
            for i in range(self.control_points_.shape[0]):
                pts_str += " -> "
                for j in range(self.control_points_.shape[1]):
                    pts_str += f" {self.control_points_[i][j]:.3f}"
            print(pts_str)

        self.optimize()

        if self.log_debug:
            pts_str = "after control_points:"
            for i in range(self.control_points_.shape[0]):
                pts_str += " -> "
                for j in range(self.control_points_.shape[1]):
                    pts_str += f" {self.control_points_[i][j]:.3f}"
            print(pts_str)
        return self.control_points_

    def optimize(self):
        self.iter_num_ = 0
        self.min_cost_ = float('inf')
        self.g_q_ = np.zeros((self.pt_num_, self.pt_dim_))
        self.variable_num_ = self.pt_dim_ * (self.pt_num_ - self.order_) if not self.end_fix_ else self.pt_dim_ * (self.pt_num_ - self.order_)
        end_idx = self.pt_num_ if self.end_fix_ else self.pt_num_ - self.order_

        opt = nlopt.opt(nlopt.LD_LBFGS, self.variable_num_)
        opt.set_min_objective(self.cost_function, self)
        opt.set_maxeval(10)
        opt.set_maxtime(0.05)
        opt.set_xtol_rel(1e-5)

        q = np.zeros(self.variable_num_)
        lb = np.zeros(self.variable_num_)
        ub = np.zeros(self.variable_num_)
        for i in range(self.order_, end_idx):
            for j in range(self.pt_dim_):
                q[self.pt_dim_ * (i - self.order_) + j] = self.control_points_[i][j]
        for i in range(self.variable_num_):
            lb[i] = q[i] - 1e5
            ub[i] = q[i] + 1e5
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

        if self.log_debug:
            print(f"q({self.variable_num_}): {q}")
            print(f"lb({self.variable_num_}): {lb}")
            print(f"ub({self.variable_num_}): {ub}")

        try:
            final_cost, x = opt.optimize(q)
            print(f"nlopt result:{int(opt.last_optimize_result())}, final_cost:{final_cost:.3f}")
        except Exception as e:
            print("nlopt exception")

        for i in range(self.order_, end_idx):
            for j in range(self.pt_dim_):
                self.control_points_[i][j] = x[self.pt_dim_ * (i - self.order_) + j]

    def cost_function(self, x, grad):
        self.iter_num_ += 1
        cost = 0.0
        self.combine_cost(x, grad, cost)
        if cost < self.min_cost_:
            self.min_cost_ = cost
            self.best_variable_ = x
        return cost

    def combine_cost(self, x, grad, cost):
        feas_pt_nums = self.variable_num_ // self.pt_dim_
        feas_g_q_str = f"\tfeas_g_q({feas_pt_nums}):"
        for i in range(self.order_):
            for j in range(self.pt_dim_):
                self.g_q_[i][j] = self.control_points_[i][j]
                if not self.end_fix_:
                    self.g_q_[self.order_ + feas_pt_nums + i][j] = self.control_points_[self.pt_num_ - self.order_ + i][j]
        for i in range(feas_pt_nums):
            feas_g_q_str += " -> "
            for j in range(self.pt_dim_):
                self.g_q_[i + self.order_][j] = x[self.pt_dim_ * i + j]
                feas_g_q_str += f" {self.g_q_[i + self.order_][j]:.3f}"
        if self.log_debug:
            print(feas_g_q_str)

        cost = 0.0
        grad[:] = 0.0
        log = ""
        def update(cost, grad, weight, f, g, sign):
            nonlocal log
            cost += weight * f
            for i in range(feas_pt_nums):
                for j in range(self.pt_dim_):
                    grad[self.pt_dim_ * i + j] += weight * g[i + self.order_][j]
            log += f" {sign}:{weight:.1f} x {f:.3f}"

        def calc_cost(func, sign, weight):
            nonlocal cost
            cost_i = 0.0
            grad_i = np.zeros((feas_pt_nums, self.pt_dim_))
            self.__getattribute__(func)(self.g_q_, cost_i, grad_i)
            update(cost, grad, weight, cost_i, grad_i, sign)

        calc_cost('calc_smoothness_cost', 'SMOOTHNESS', self.conf_.w_smoothness)
        calc_cost('calc_distance_cost', 'DISTANCE', self.conf_.w_distance)
        calc_cost('calc_guide_cost', 'GUIDE', self.conf_.w_guide)
        calc_cost('calc_endpoint_cost', 'ENDPOINT', self.conf_.w_endpoint)

        print(f"iter({self.iter_num_}) var_num:{self.variable_num_}, cost:{cost:.3f}" + log)

    def calc_smoothness_cost(self, q, cost, gradient):
        s_str = f"\nCalcSmoothnessCost cost:{cost:.3f}"
        for i in range(self.pt_num_ - self.order_):
            jerk = q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]
            cur_cost = jerk.dot(jerk)
            s_str += f"\n\tcp:{i}-{i + 3}, cost:{cur_cost:.3f}, jerk:({jerk[0]:.3f},{jerk[1]:.3f})"
            cost += cur_cost
            temp_j = 2.0 * jerk
            gradient[i + 0] += -temp_j
            gradient[i + 1] += 3.0 * temp_j
            gradient[i + 2] += -3.0 * temp_j
            gradient[i + 3] += temp_j
        if self.log_debug:
            print(s_str)

    def calc_distance_cost(self, q, cost, gradient):
        d_str = f"\nCalcDistanceCost cost:{cost:.3f}, safe_dist:{self.conf_.safe_dist:.3f}"
        end_idx = self.pt_num_ if self.end_fix_ else self.pt_num_ - self.order_
        for i in range(self.order_, end_idx):
            dist = self.om_.get_point_min_distance_to_obstacle((q[i][0], q[i][1]))
            dist_grad_ad2 = self.om_.get_point_gradient_direction((q[i][0], q[i][1]))
            dist_grad = np.array(dist_grad_ad2)
            dist_grad /= np.linalg.norm(dist_grad)
            d_str += f"\n\tcp:{i}, dist:{dist:.3f}, pose:({q[i][0]:.3f}, {q[i][1]:.3f}), dist_grad:({dist_grad_ad2[0]}, {dist_grad_ad2[1]})"
            if dist < self.conf_.safe_dist:
                cost += (dist - self.conf_.safe_dist) ** 2
                gradient[i] += 2.0 * (dist - self.conf_.safe_dist) * dist_grad
        if self.log_debug:
            print(d_str)

    def calc_guide_cost(self, q, cost, gradient):
        g_str = f"\nCalcGuideCost cost:{cost:.3f}"
        for i in range(self.order_, self.pt_num_ - self.order_):
            delta = q[i] - self.guide_pts_[i]
            cur_cost = delta.dot(delta)
            g_str += f"\n\tcp:{i}, delta:({delta[0]:.3f}, {delta[1]:.3f}), dis:{cur_cost:.3f}"
            cost += cur_cost
            gradient[i] += 2 * delta
        if self.log_debug:
            print(g_str)

    def calc_endpoint_cost(self, q, cost, gradient):
        if not self.end_fix_:
            return
        e_str = f"\nCalcEndpointCost cost:{cost:.3f}"
        cp_3 = self.control_points_[self.pt_num_ - 3]
        cp_2 = self.control_points_[self.pt_num_ - 2]
        cp_1 = self.control_points_[self.pt_num_ - 1]
        ref = 1 / 6.0 * (cp_3 + 4 * cp_3 + cp_1)
        q_3 = q[self.pt_num_ - 3]
        q_2 = q[self.pt_num_ - 2]
        q_1 = q[self.pt_num_ - 1]
        opt = 1 / 6.0 * (q_3 + 4 * q_2 + q_1)
        dq = opt - ref
        cur_cost = dq.dot(dq)
        e_str += f"\n\tdq:({dq[0]:.3f}, {dq[1]:.3f}), dis:{cur_cost:.3f}"
        cost += cur_cost
        gradient[self.pt_num_ - 3] += 2 * dq * (1 / 6.0)
        gradient[self.pt_num_ - 2] += 2 * dq * (4 / 6.0)
        gradient[self.pt_num_ - 1] += 2 * dq * (1 / 6.0)
        if self.log_debug:
            print(e_str)