import sympy as sp


class RootFinder:
    def __init__(self, func):
        self.func = func
        self.x = sp.symbols('x')
        self.func_prime = sp.diff(func, self.x)

    def bisection_method(self, a, b, tol=1e-6, max_iter=1000):
        f = sp.lambdify(self.x, self.func, 'numpy')
        if f(a) * f(b) > 0:
            print("The function has the same sign at the endpoints, unable to use bisection method.")
            return None

        iter_count = 0
        while (b - a) / 2 > tol and iter_count < max_iter:
            c = (a + b) / 2
            if f(c) == 0:
                break
            elif f(a) * f(c) < 0:
                b = c
            else:
                a = c
            iter_count += 1

        return (a + b) / 2

    def newton_method(self, x0, tol=1e-6, max_iter=1000, max_value=1e20):
        f = sp.lambdify(self.x, self.func, 'numpy')
        f_prime = sp.lambdify(self.x, self.func_prime, 'numpy')

        x = x0
        iter_count = 0
        while abs(f(x)) > tol and iter_count < max_iter:
            f_val = f(x)
            f_prime_val = f_prime(x)

            if abs(f_prime_val) < 1e-12:
                print("Derivative is too small, Newton's method may fail.")
                return None

            x_new = x - f_val / f_prime_val
            if abs(x_new) > max_value:
                print("Value too large, stopping iteration.")
                return None

            x = x_new
            iter_count += 1

        return x


def test():
    x = sp.symbols('x')
    func = x ** 3 - x - 2

    root_finder = RootFinder(func)
    print(f"The derivative of the function is: {root_finder.func_prime}")

    a, b = 1, 2
    root_bisection = root_finder.bisection_method(a, b)
    print(f"The root found by bisection method is: {root_bisection}")

    x0 = 1.5
    root_newton = root_finder.newton_method(x0)
    print(f"The root found by Newton's method is: {root_newton}")


if __name__ == "__main__":
    test()
