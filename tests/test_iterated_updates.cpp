#include <cmath>

#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/srif.hpp"

int main() {
  const double x_true = 1.0;
  const double y = x_true * x_true;
  const double sigma = 0.2;
  const double sw = 1.0 / sigma;

  Eigen::VectorXd x0(1);
  x0(0) = 2.5;
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Constant(1, 1, 1.0);

  auto h = [](const Eigen::VectorXd& x) { return x(0) * x(0); };
  auto j = [](const Eigen::VectorXd& x) {
    Eigen::RowVectorXd H(1);
    H(0) = 2.0 * x(0);
    return H;
  };

  bierman::filters::KalmanState kf_single{x0, P0};
  bierman::filters::KalmanJoseph::update_scalar(kf_single, j(x0), y - h(x0), sw);

  bierman::filters::KalmanState kf_iter{x0, P0};
  bierman::filters::KalmanJoseph::update_scalar_iterated(kf_iter, y, sw, h, j, 8, 1e-12);

  const double r_single = std::abs(y - h(kf_single.x));
  const double r_iter = std::abs(y - h(kf_iter.x));
  if (r_iter > r_single + 1e-12) {
    spdlog::error("Iterated Kalman did not reduce residual: single={} iter={}", r_single, r_iter);
    return 1;
  }

  Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(1, 1));
  bierman::filters::SRIFState srif{F0.llt().matrixU(), F0.llt().matrixU() * x0};

  Eigen::MatrixXd sqrtW(1, 1);
  sqrtW(0, 0) = sw;
  Eigen::VectorXd yv(1);
  yv(0) = y;

  auto s_single = bierman::filters::SRIF::update_householder(
      srif,
      j(x0),
      yv - Eigen::VectorXd::Constant(1, h(x0)) + j(x0) * x0,
      sqrtW);
  auto s_iter = bierman::filters::SRIF::update_householder_iterated(
      srif,
      yv,
      sqrtW,
      x0,
      [&](const Eigen::VectorXd& x) {
        Eigen::VectorXd out(1);
        out(0) = h(x);
        return out;
      },
      [&](const Eigen::VectorXd& x) {
        Eigen::MatrixXd H(1, 1);
        H(0, 0) = 2.0 * x(0);
        return H;
      },
      8,
      1e-12);

  const double rs_single = std::abs(y - h(s_single.x));
  const double rs_iter = std::abs(y - h(s_iter.x));
  if (rs_iter > rs_single + 1e-12) {
    spdlog::error("Iterated SRIF did not reduce residual: single={} iter={}", rs_single, rs_iter);
    return 1;
  }

  return 0;
}
