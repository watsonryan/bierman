#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::linalg {

struct UDFactors {
  Eigen::MatrixXd U;  // Unit upper triangular
  Eigen::VectorXd d;  // Diagonal elements
};

inline UDFactors ud_factorize(const Eigen::Ref<const Eigen::MatrixXd>& Pin) {
  if (Pin.rows() != Pin.cols()) {
    throw std::invalid_argument("ud_factorize expects a square matrix");
  }

  const Eigen::Index n = Pin.rows();
  Eigen::MatrixXd P = Pin;
  Eigen::MatrixXd U = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd d = Eigen::VectorXd::Zero(n);

  for (Eigen::Index jj = n - 1; jj >= 1; --jj) {
    const double dj = P(jj, jj);
    if (dj <= 0.0) {
      throw std::runtime_error("ud_factorize requires a positive-definite matrix");
    }
    d(jj) = dj;
    const double alpha = 1.0 / dj;

    for (Eigen::Index kk = 0; kk <= jj - 1; ++kk) {
      const double beta = P(kk, jj);
      U(kk, jj) = alpha * beta;
      for (Eigen::Index ii = 0; ii <= kk; ++ii) {
        P(ii, kk) -= beta * U(ii, jj);
      }
    }
  }

  if (P(0, 0) <= 0.0) {
    throw std::runtime_error("ud_factorize requires a positive-definite matrix");
  }
  d(0) = P(0, 0);

  return {U, d};
}

inline Eigen::MatrixXd ud_reconstruct(const Eigen::Ref<const Eigen::MatrixXd>& U,
                                      const Eigen::Ref<const Eigen::VectorXd>& d) {
  if (U.rows() != U.cols() || U.rows() != d.size()) {
    throw std::invalid_argument("ud_reconstruct dimension mismatch");
  }
  return U * d.asDiagonal() * U.transpose();
}

}  // namespace bierman::linalg
