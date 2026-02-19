#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Core>

namespace bierman::linalg {

inline bool chol_rank1_update_lower(Eigen::Ref<Eigen::MatrixXd> L,
                                    const Eigen::Ref<const Eigen::VectorXd>& v,
                                    double beta) {
  if (L.rows() != L.cols() || L.rows() != v.size()) {
    throw std::invalid_argument("chol_rank1_update_lower dimension mismatch");
  }
  if (beta != 1.0 && beta != -1.0) {
    throw std::invalid_argument("chol_rank1_update_lower beta must be +1 or -1");
  }

  const Eigen::Index n = L.rows();
  Eigen::VectorXd w = v;
  Eigen::MatrixXd L2 = Eigen::MatrixXd::Zero(n, n);

  double b = 1.0;
  for (Eigen::Index j = 0; j < n; ++j) {
    const double ljj = L(j, j);
    const double wj = w(j);

    const double diag_arg = ljj * ljj + beta * wj * wj / b;
    if (diag_arg <= 0.0) {
      return false;
    }

    const double l2jj = std::sqrt(diag_arg);
    L2(j, j) = l2jj;

    const double gamma = ljj * ljj * b + beta * wj * wj;
    if (gamma == 0.0) {
      return false;
    }

    for (Eigen::Index k = j + 1; k < n; ++k) {
      w(k) = w(k) - wj * L(k, j) / ljj;
      L2(k, j) = (l2jj * L(k, j) / ljj) + (l2jj * beta * wj * w(k) / gamma);
    }

    b = b + beta * wj * wj / (ljj * ljj);
    if (b <= 0.0) {
      return false;
    }
  }

  L = L2;
  return true;
}

}  // namespace bierman::linalg
