#include "polynomial_fitting.h"

#include <iostream>

PolynomialFitting::PolynomialFitting(const std::vector<double>& x,
                                     const std::vector<double>& y, int degree)
    : x_data(x), y_data(y), degree(degree) {
  int n = x_data.size();
  vandermonde_matrix = Eigen::MatrixXd(n, degree + 1);

  // 构建 Vandermonde 矩阵
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= degree; ++j) {
      vandermonde_matrix(i, j) = std::pow(x_data[i], j);
    }
  }
}

Eigen::VectorXd PolynomialFitting::FitPolynomial() {
  Eigen::VectorXd y_vec(y_data.size());
  for (size_t i = 0; i < y_data.size(); ++i) {
    y_vec(i) = y_data[i];
  }

  // 使用最小二乘法求解 (V^T * V) * coefficients = V^T * y
  coefficients = (vandermonde_matrix.transpose() * vandermonde_matrix)
                     .ldlt()
                     .solve(vandermonde_matrix.transpose() * y_vec);

  return coefficients;
}

double PolynomialFitting::Evaluate(double x) {
  double result = 0.0;
  for (int i = 0; i <= degree; ++i) {
    result += coefficients(i) * std::pow(x, i);
  }
  return result;
}
