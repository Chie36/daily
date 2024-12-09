import numpy as np


class ILqr:
    def __init__(self):
        self.n_state = 1
        self.n_control = 1

        self.N = 3
        self.x0 = np.ones(self.n_state)

        self.Q = np.eye(self.n_state)
        self.R = np.eye(self.n_control)

        self.x_traj = np.zeros((self.N + 1, self.n_state))
        self.u_traj = np.zeros((self.N, self.n_control))

    def system_dynamics(self, x, u):
        return x + np.sin(u)

    def cost_function_lqr(self, x, u):
        # L(x,u)
        return 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u)

    def l_u(self, u):
        return self.R @ u

    def l_uu(self):
        return self.R

    def l_x(self, x):
        return self.Q @ x

    def l_xx(self):
        return self.Q

    def compute_initial_trajectory(self, x0, u_traj):
        x_traj = np.zeros((self.N + 1, self.n_state))
        x_traj[0] = x0
        for k in range(self.N):
            x_traj[k + 1] = self.system_dynamics(x_traj[k], u_traj[k])
        return x_traj

    def solve_ilqr(self):
        # Max iterations
        iterations = 50

        # Tolerance
        eps_u = 1e-3
        eps_x = 1e-4
        eps_J = 1e-4

        pre_cost = np.inf

        # 1.Backward pass
        for i in range(iterations):
            V_x = np.zeros(self.n_state + 1)
            V_xx = np.zeros((self.n_state + 1, self.n_state + 1))

    def output_traj(self):
        for x, u in zip(self.x_traj[:-1], self.u_traj):
            formatted_x = np.array2string(
                x, formatter={"float_kind": lambda x: f"{x:.4f}"}
            )
            formatted_u = np.array2string(
                u, formatter={"float_kind": lambda x: f"{x:.4f}"}
            )
            print(f"X({formatted_x})  ---U({formatted_u})--->", end="  ")
        formatted_x = np.array2string(
            self.x_traj[-1], formatter={"float_kind": lambda x: f"{x:.4f}"}
        )
        print(f"X({formatted_x})")


if __name__ == "__main__":
    lqr = Lqr()
    lqr.solve_lqr()
    lqr.apply_lqr_control()
    lqr.output_traj()
