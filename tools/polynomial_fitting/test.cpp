#include <iostream>
#include <vector>

#include "polynomial_fitting.h"

int main() {
  std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y_data = {1.2, 2.8, 3.6, 5.0, 7.2};
  int degree = 2;

  PolynomialFitting pf(x_data, y_data, degree);
  Eigen::VectorXd coefficients = pf.FitPolynomial();
  std::cout << "Fitted polynomial coefficients: \n["
            << coefficients.transpose().reverse() << "]" << std::endl;
  for (double x = 1.0; x <= 5.0; x += 1.0) {
    std::cout << "P(" << x << ") = " << pf.Evaluate(x) << std::endl;
  }

  return 0;
}
