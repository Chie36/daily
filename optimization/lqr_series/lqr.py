import numpy as np


class Lqr:
    def __init__(self):
        self.n_state = 1
        self.n_control = 1

        self.N = 3
        self.x0 = np.ones(self.n_state)

        self.A = np.eye(1)  # np.zeros((self.n_state, self.n_state))
        self.B = np.eye(1)  # np.zeros((self.n_control, self.n_state))

        self.Q = np.eye(self.n_state)
        self.R = np.eye(self.n_control)

        self.P_seq = np.zeros((self.N + 1, self.n_state, self.n_state))
        self.K_seq = np.zeros((self.N, self.n_control, self.n_state))

        self.x_traj = np.zeros((self.N + 1, self.n_state))
        self.u_traj = np.zeros((self.N, self.n_control))

    def system_dynamics(self, x, u):
        # f(x,u)
        return self.A @ x + self.B @ u

    def cost_function_lqr(self, x, u):
        # L(x,u)
        return 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u)

    def solve_lqr(self):
        # 1.Backward pass
        # The P matrix at the last step is the state cost matrix
        self.P_seq[self.N] = self.Q
        for k in range(self.N - 1, -1, -1):
            # Riccati equation computation
            S = self.R + self.B.T @ self.P_seq[k + 1] @ self.B
            self.P_seq[k] = (
                self.Q
                + self.A.T @ self.P_seq[k + 1] @ self.A
                - self.A.T
                @ self.P_seq[k + 1]
                @ self.B
                @ np.linalg.inv(S)
                @ self.B.T
                @ self.P_seq[k + 1]
                @ self.A
            )
            self.K_seq[k] = np.linalg.inv(S) @ self.B.T @ self.P_seq[k + 1] @ self.A

        # 2.Forward pass
        # Set the initial state
        self.x_traj[0] = self.x0
        for k in range(self.N):
            # Apply LQR control law
            self.u_traj[k] = -self.K_seq[k] @ self.x_traj[k]
            # State update
            self.x_traj[k + 1] = self.system_dynamics(self.x_traj[k], self.u_traj[k])

        # 3.Output trajectory
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
    lqr = Lqr()
    lqr.solve_lqr()
