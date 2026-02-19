#pragma once

#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace bierman::internal {

// Internal utility (not part of stable public API).
inline Eigen::MatrixXd robust_chol_with_jitter(const Eigen::Ref<const Eigen::MatrixXd>& Pin,
                                               const char* err_msg) {
  Eigen::MatrixXd P = 0.5 * (Pin + Pin.transpose());
  const Eigen::Index n = P.rows();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  double jitter = 1e-12;
  for (int k = 0; k < 12; ++k) {
    Eigen::LLT<Eigen::MatrixXd> llt(P + jitter * I);
    if (llt.info() == Eigen::Success) {
      return llt.matrixL();
    }
    jitter *= 10.0;
  }
  throw std::runtime_error(err_msg);
}

}  // namespace bierman::internal
