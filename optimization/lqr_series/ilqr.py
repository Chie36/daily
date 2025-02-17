import numpy as np


class ILqr:
    def __init__(self, x_init_traj, u_init_traj):
        self.n_state = 1
        self.n_control = 1

        self.N = u_init_traj.shape[0]
        self.x0 = np.ones(self.n_state)

        self.Q = np.eye(self.n_state)
        self.R = np.eye(self.n_control)

        self.x_init_traj = x_init_traj
        self.u_init_traj = u_init_traj
        self.x_iter_traj = self.x_init_traj
        self.u_iter_traj = self.u_init_traj
        self.x_traj = self.x_init_traj
        self.u_traj = self.u_init_traj

        self.F_x_seq = np.zeros((self.N + 1, self.n_state, self.n_state))
        self.F_u_seq = np.zeros((self.N, self.n_control, self.n_state))

        self.L_u_seq = np.zeros((self.N, self.n_control, 1))
        self.L_x_seq = np.zeros((self.N + 1, self.n_state, 1))
        self.L_uu_seq = np.zeros((self.N, self.n_control, self.n_control))
        self.L_xx_seq = np.zeros((self.N + 1, self.n_state, self.n_state))
        self.L_ux_seq = np.zeros((self.N + 1, self.n_control, self.n_state))

        self.V_x_seq = np.zeros((self.N + 1, self.n_state, 1))
        self.V_xx_seq = np.zeros((self.N + 1, self.n_state, self.n_state))

        self.K_seq = np.zeros((self.N, self.n_control, self.n_state))
        self.d_seq = np.zeros((self.N, self.n_control, 1))

    def system_dynamics(self, x, u):
        # f(x,u)
        return x + u

    def cost_function_ilqr(self, x, u):
        # L(x,u)
        return 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u)

    def get_system_derivative(self):
        for k in range(self.N):
            F_u = np.eye(self.n_control)
            self.F_u_seq[k] = F_u
        for k in range(self.N + 1):
            F_x = np.eye(self.n_state)
            self.F_x_seq[k] = F_x

    def get_cost_derivative(self):
        for k in range(self.N):
            L_u = self.R @ self.u_iter_traj[k]
            L_uu = self.R
            self.L_u_seq[k] = L_u.reshape(self.n_control, 1)
            self.L_uu_seq[k] = L_uu

        for k in range(self.N + 1):
            dx = self.x_iter_traj[k] - self.x_init_traj[k]
            L_x = self.Q @ dx
            L_xx = self.Q
            self.L_x_seq[k] = L_x.reshape(self.n_state, 1)
            self.L_xx_seq[k] = L_xx

    def solve_ilqr(self):
        iterations = 50
        pre_cost = np.inf
        lamb = 0.5
        cost_eps = 1e-2

        for i in range(iterations):
            # 1.Backward pass
            self.get_system_derivative()
            self.get_cost_derivative()

            V_x = self.L_x_seq[-1]
            V_xx = self.L_xx_seq[-1]
            for k in range(self.N - 1, -1, -1):
                Q_x = self.L_x_seq[k] + self.F_x_seq[k].T @ V_x
                Q_u = self.L_u_seq[k] + self.F_u_seq[k].T @ V_x
                Q_xx = self.L_xx_seq[k] + self.F_x_seq[k].T @ V_xx @ self.F_x_seq[k]
                Q_uu = self.L_uu_seq[k] + self.F_u_seq[k].T @ V_xx @ self.F_u_seq[k]
                Q_ux = self.L_ux_seq[k] + self.F_u_seq[k].T @ V_xx @ self.F_x_seq[k]

                # Q_uu[0, 0] += lamb
                self.K_seq[k] = -np.linalg.inv(Q_uu) @ Q_ux
                self.d_seq[k] = -np.linalg.inv(Q_uu) @ Q_u

                V_x = (
                    Q_x
                    + self.K_seq[k].T @ Q_uu @ self.d_seq[k]
                    + self.K_seq[k].T @ Q_u
                    + Q_ux.T @ self.d_seq[k]
                )
                V_xx = (
                    Q_xx
                    + self.K_seq[k].T @ Q_uu @ self.K_seq[k]
                    + self.K_seq[k].T @ Q_ux
                    + Q_ux.T @ self.K_seq[k]
                )

            # 2.Forward pass
            x_traj = np.zeros((self.N + 1, self.n_state))
            u_traj = np.zeros((self.N, self.n_control))
            x_traj[0] = self.x_iter_traj[0]
            for k in range(self.N):
                x_err = x_traj[k] - self.x_iter_traj[k]
                u = self.u_iter_traj[k] + self.K_seq[k] @ x_err + self.d_seq[k]
                x = self.system_dynamics(x_traj[k], u)

                u_traj[k] = u
                x_traj[k + 1] = x

            # 3. Calc cost
            cost = 0
            for k in range(self.N):
                dx = self.x_iter_traj[k] - self.x_init_traj[k]
                cost += self.cost_function_ilqr(
                    dx.reshape(self.n_state, 1),
                    self.x_init_traj[-1].reshape(self.n_control, 1),
                )[0, 0]
            dx_final = self.x_iter_traj[-1] - self.x_init_traj[-1]
            cost += self.cost_function_ilqr(
                dx_final.reshape(self.n_state, 1), np.zeros((self.n_control, 1))
            )[0, 0]

            # 4.Update trajectory
            self.u_iter_traj = u_traj
            self.x_iter_traj = x_traj

            # 5. Check converged
            cost_err = abs(pre_cost - cost)
            print(
                f"Iteration {i}: pre_cost({pre_cost:.4f}), cost({cost:.4f}), cost_err({cost_err:.4f})"
            )

            if cost_err < cost_eps:
                print(f"Converged at iteration {i}")
                self.x_traj = self.u_iter_traj
                self.u_traj = self.x_iter_traj
                break

            pre_cost = cost

        self.output_traj()

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
    x_init_traj = np.array([[1.20], [0.48], [0.18], [0.09]])
    u_init_traj = np.array([[-0.51], [-0.16], [-0.08]])
    lqr = ILqr(x_init_traj, u_init_traj)
    lqr.solve_ilqr()
