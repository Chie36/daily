#ifndef POLYNOMIAL_FITTING_H
#define POLYNOMIAL_FITTING_H

#include <Eigen/Dense>
#include <vector>

class PolynomialFitting {
 public:
  PolynomialFitting(const std::vector<double>& x, const std::vector<double>& y,
                    int degree);
  Eigen::VectorXd FitPolynomial();
  double Evaluate(double x);

 private:
  std::vector<double> x_data;
  std::vector<double> y_data;
  int degree;
  Eigen::MatrixXd vandermonde_matrix;
  Eigen::VectorXd coefficients;
};

#endif  // POLYNOMIAL_FITTING_H
