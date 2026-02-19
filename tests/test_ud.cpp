#include <cmath>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/linalg/ud.hpp"

int main() {
  constexpr int n = 8;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P = A * A.transpose() + 1e-2 * Eigen::MatrixXd::Identity(n, n);

  const auto factors = bierman::linalg::ud_factorize(P);
  const Eigen::MatrixXd Pr = bierman::linalg::ud_reconstruct(factors.U, factors.d);

  const double rel_err = (P - Pr).norm() / P.norm();
  if (!std::isfinite(rel_err) || rel_err > 1e-10) {
    spdlog::error("UD reconstruction relative error too high: {}", rel_err);
    return 1;
  }

  if ((factors.U.diagonal().array() - 1.0).abs().maxCoeff() > 1e-12) {
    spdlog::error("U diagonal is not unit.");
    return 1;
  }

  if ((factors.d.array() <= 0.0).any()) {
    spdlog::error("D has non-positive diagonal entries.");
    return 1;
  }

  return 0;
}
