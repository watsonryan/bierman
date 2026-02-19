#include <cmath>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"

int main() {
  constexpr int n = 6;

  Eigen::MatrixXd M = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P0 = M * M.transpose() + 1e-2 * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Eigen::RowVectorXd A = Eigen::RowVectorXd::Random(n);

  const double delta = 0.37;
  const double sigma_obs = 2.0;
  const double sw = 1.0 / sigma_obs;

  bierman::filters::KalmanState s{x0, P0};
  bierman::filters::KalmanJoseph::update_scalar(s, A, delta, sw);

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd R = Eigen::MatrixXd::Constant(1, 1, sigma_obs * sigma_obs);
  const Eigen::MatrixXd H = A;
  const Eigen::MatrixXd S = H * P0 * H.transpose() + R;
  const Eigen::MatrixXd K = (P0 * H.transpose()) / S(0, 0);
  const Eigen::VectorXd x_ref = x0 + K * delta;
  const Eigen::MatrixXd IKH = I - K * H;
  const Eigen::MatrixXd P_ref = IKH * P0 * IKH.transpose() + K * R * K.transpose();

  const double x_err = (s.x - x_ref).norm();
  const double p_rel_err = (s.P - P_ref).norm() / P_ref.norm();

  if (!std::isfinite(x_err) || x_err > 1e-10) {
    spdlog::error("KalmanJoseph state mismatch: {}", x_err);
    return 1;
  }
  if (!std::isfinite(p_rel_err) || p_rel_err > 1e-10) {
    spdlog::error("KalmanJoseph covariance mismatch: {}", p_rel_err);
    return 1;
  }

  Eigen::LLT<Eigen::MatrixXd> llt(s.P);
  if (llt.info() != Eigen::Success) {
    spdlog::error("KalmanJoseph covariance is not SPD after update.");
    return 1;
  }

  return 0;
}
