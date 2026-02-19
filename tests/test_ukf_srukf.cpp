#include <cmath>

#include <spdlog/spdlog.h>

#include "bierman/filters/srukf.hpp"
#include "bierman/filters/ukf.hpp"

int main() {
  constexpr int n = 5;
  constexpr int m = 3;

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P0 = A * A.transpose() + 0.5 * Eigen::MatrixXd::Identity(n, n);

  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
  F(0, 2) = 0.1;
  F(1, 3) = -0.05;
  F(4, 0) = 0.03;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, n);
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;
  H(2, 4) = 1.0;

  Eigen::MatrixXd Q = 1e-3 * Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd R = 5e-3 * Eigen::MatrixXd::Identity(m, m);

  auto prop = [&F](const Eigen::VectorXd& x) { return F * x; };
  auto meas = [&H](const Eigen::VectorXd& x) { return H * x; };

  bierman::filters::UTParams p{0.2, 0.0, 2.0};
  bierman::filters::UKFState ukf{x0, P0};
  auto up = bierman::filters::predict(ukf, Q, p, prop);
  Eigen::VectorXd y = meas(up.state.x);
  y(0) += 0.01;
  y(2) -= 0.02;
  auto uo = bierman::filters::update(up, y, R, meas);

  bierman::filters::SRUKFState sr{x0, P0.llt().matrixL()};
  Eigen::MatrixXd Lq = Q.llt().matrixL();
  Eigen::MatrixXd Lr = R.llt().matrixL();
  auto sp = bierman::filters::SRUKF::predict(sr, Lq, p, prop);
  auto so = bierman::filters::SRUKF::update(sp, y, Lr, meas);

  const double xerr = (uo.x - so.x).norm();
  const Eigen::MatrixXd Ps = so.L * so.L.transpose();
  const double perr = (uo.P - Ps).norm() / uo.P.norm();

  if (!std::isfinite(xerr) || !std::isfinite(perr) || xerr > 5e-3 || perr > 5e-2) {
    spdlog::error("UKF/SRUKF mismatch xerr={} perr={}", xerr, perr);
    return 1;
  }

  return 0;
}
