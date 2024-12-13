import numpy as np

# Custom Parameters
max_iter = 10000
tol = 1e-6


class BasicSolver:
    def __init__(self, x, y, params):
        self.x_data = x
        self.y_data = y
        self.init_params = params

    def model(self, params):
        a, b = params
        return a * np.exp(b * self.x_data)

    def residuals(self, params):
        return self.model(params) - self.y_data

    def objective_function(self, params):
        res = self.residuals(params)
        return 0.5 * np.sum(res**2)

    def gradient(self, params):
        a, b = params
        res = self.residuals(params)
        grad_a = np.sum(res * np.exp(b * self.x_data))
        grad_b = np.sum(res * a * self.x_data * np.exp(b * self.x_data))
        return np.array([grad_a, grad_b])

    def jacobian(self, params):
        # g = J.T @ res
        a, b = params
        J = np.zeros((len(self.x_data), len(params)))
        J[:, 0] = np.exp(b * self.x_data)
        J[:, 1] = a * self.x_data * np.exp(b * self.x_data)
        return J

    def hessian(self, params):
        a, b = params
        H = np.zeros((2, 2))
        H[0, 0] = np.sum(np.exp(2 * b * self.x_data))
        H[0, 1] = H[1, 0] = np.sum(self.x_data * np.exp(2 * b * self.x_data))
        H[1, 1] = np.sum(a * self.x_data**2 * np.exp(2 * b * self.x_data))
        return H

    def gradient_descent_method(self):
        print("-> Gradient Descent Method:")
        learning_rate = 0.1
        params = np.array(self.init_params)

        for i in range(max_iter):
            gradient = self.gradient(params)

            params_new = params - learning_rate * gradient

            if np.linalg.norm(params_new - params) < tol:
                print(f"Converged after {i + 1} iterations.")
                return params_new

            params = params_new

        print("Max iterations exceeded. No solution found.")
        return params

    def newton_method(self):
        print("-> Newton Method:")
        params = np.array(self.init_params)

        for i in range(max_iter):
            gradient = self.gradient(params)
            hessian = self.hessian(params)

            if np.linalg.det(hessian) == 0:
                print("Hessian matrix is singular.")
                return None

            params_new = params - np.linalg.inv(hessian) @ gradient

            if np.linalg.norm(params_new - params) < tol:
                print(f"Converged after {i + 1} iterations.")
                return params_new

            params = params_new
        print("Max iterations exceeded. No solution found.")
        return params

    def gauss_newton_method(self):
        print("-> Gauss Newton Method:")
        params = np.array(self.init_params)

        for i in range(max_iter):
            res = self.residuals(params)
            J = self.jacobian(params)

            params_new = params - np.linalg.inv(J.T @ J) @ (J.T @ res)

            if np.linalg.norm(params_new - params) < tol:
                print(f"Converged after {i + 1} iterations.")
                return params_new

            params = params_new

        print("Max iterations exceeded. No solution found.")
        return params

    def levenberg_marquardt_method(self):
        print("-> Levenburg Marquardt Method:")
        lamb = 0.01
        params = np.array(self.init_params)

        for i in range(max_iter):
            res = self.residuals(params)
            J = self.jacobian(params)

            params_new = params - np.linalg.inv(
                J.T @ J + lamb * np.eye(len(params))
            ) @ (J.T @ res)

            if np.linalg.norm(params_new - params) < tol:
                print(f"Converged after {i + 1} iterations.")
                return params_new

            res_new = self.residuals(params_new)
            if np.sum(res_new**2) < np.sum(res**2):
                params = params_new
                lamb /= 10
            else:
                lamb *= 10

        print("Max iterations exceeded. No solution found.")
        return params


if __name__ == "__main__":
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = 2 * np.exp(-1 * x_data)
    initial_guess = [4.0, -2.0]
    solver = BasicSolver(x_data, y_data, initial_guess)

    def test(func):
        solution = func()
        if solution is not None:
            print(f"Fitted parameters: a = {solution[0]:.4f}, b = {solution[1]:.4f}")
        else:
            print("No solution found.")

    test(solver.gradient_descent_method)
    test(solver.newton_method)
    test(solver.gauss_newton_method)
    test(solver.levenberg_marquardt_method)
