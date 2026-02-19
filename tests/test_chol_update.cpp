#include <cmath>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/linalg/chol_update.hpp"

int main() {
  constexpr int n = 7;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P = A * A.transpose() + Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd v = Eigen::VectorXd::Random(n);

  Eigen::LLT<Eigen::MatrixXd> llt(P);
  if (llt.info() != Eigen::Success) {
    spdlog::error("Initial Cholesky failed.");
    return 1;
  }

  Eigen::MatrixXd L = llt.matrixL();
  Eigen::MatrixXd L_update = L;
  if (!bierman::linalg::chol_rank1_update_lower(L_update, v, +1.0)) {
    spdlog::error("Rank-one update unexpectedly failed.");
    return 1;
  }

  const Eigen::MatrixXd P_update_expected = P + v * v.transpose();
  const Eigen::MatrixXd P_update_actual = L_update * L_update.transpose();
  const double update_rel_err = (P_update_actual - P_update_expected).norm() / P_update_expected.norm();
  if (!std::isfinite(update_rel_err) || update_rel_err > 1e-10) {
    spdlog::error("Rank-one update error too high: {}", update_rel_err);
    return 1;
  }

  Eigen::MatrixXd L_downdate = L_update;
  if (!bierman::linalg::chol_rank1_update_lower(L_downdate, v, -1.0)) {
    spdlog::error("Rank-one downdate unexpectedly failed.");
    return 1;
  }

  const Eigen::MatrixXd P_downdate_actual = L_downdate * L_downdate.transpose();
  const double downdate_rel_err = (P_downdate_actual - P).norm() / P.norm();
  if (!std::isfinite(downdate_rel_err) || downdate_rel_err > 1e-10) {
    spdlog::error("Rank-one downdate error too high: {}", downdate_rel_err);
    return 1;
  }

  Eigen::MatrixXd L_fail = L;
  const Eigen::VectorXd v_fail = 2.5 * L.col(0);
  if (bierman::linalg::chol_rank1_update_lower(L_fail, v_fail, -1.0)) {
    spdlog::error("Downdate should have failed but succeeded.");
    return 1;
  }

  return 0;
}
