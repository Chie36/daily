import numpy as np
import matplotlib.pyplot as plt


class PolynomialFitting:
    def __init__(self, x_data, y_data, degree):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.degree = degree
        self.coefficients = None

    def fit_polynomial(self):
        # 使用 numpy.polyfit 进行多项式拟合，degree 表示多项式的阶数
        self.coefficients = np.polyfit(self.x_data, self.y_data, self.degree)
        return self.coefficients

    def evaluate(self, x):
        # 使用拟合的多项式系数来计算给定 x 值的多项式值
        poly = np.poly1d(self.coefficients)
        return poly(x)


if __name__ == "__main__":
    x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_data = [1.2, 2.8, 3.6, 5.0, 7.2]
    degree = 2

    pf = PolynomialFitting(x_data, y_data, degree)
    coefficients = pf.fit_polynomial()
    print("Fitted polynomial coefficients:")
    print(coefficients)
    for x in np.arange(1.0, 6.0, 1.0):
        print(f"P({x}) = {pf.evaluate(x)}")

    # 可视化拟合结果
    x_fit = np.linspace(1, 5, 100)
    y_fit = pf.evaluate(x_fit)
    plt.scatter(x_data, y_data, color="blue", label="Data points")
    plt.plot(x_fit, y_fit, color="red", label=f"Fitted Polynomial (degree={degree})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Polynomial Fitting")
    plt.show()
