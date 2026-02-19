#include <cmath>

#include <spdlog/spdlog.h>

#include "bierman/filters/srukf.hpp"

int main() {
  constexpr int n = 8;
  constexpr int m = 4;
  constexpr int steps = 300;

  Eigen::VectorXd x = Eigen::VectorXd::Random(n);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P = A * A.transpose() + 0.05 * Eigen::MatrixXd::Identity(n, n);

  bierman::filters::SRUKFState s{x, P.llt().matrixL()};
  bierman::filters::UTParams p{0.15, 0.0, 2.0};
  bierman::filters::SRUKFWorkspace ws;
  bierman::filters::SRUKFDiagnostics diag;

  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(n, n);
  F(0, 2) = 0.05;
  F(2, 3) = -0.04;
  F(4, 1) = 0.03;
  F(6, 7) = -0.02;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m, n);
  H(0, 0) = 1.0;
  H(1, 2) = 1.0;
  H(2, 4) = 1.0;
  H(3, 6) = 1.0;

  Eigen::MatrixXd Q = 1e-3 * Eigen::MatrixXd::Identity(n, n);
  Eigen::MatrixXd R = 1e-2 * Eigen::MatrixXd::Identity(m, m);

  auto prop = [&F](const Eigen::VectorXd& xin) { return F * xin; };
  auto meas = [&H](const Eigen::VectorXd& xin) { return H * xin; };

  for (int k = 0; k < steps; ++k) {
    Eigen::MatrixXd Lq = Q.llt().matrixL();
    Eigen::MatrixXd Lr = R.llt().matrixL();

    auto pred = bierman::filters::SRUKF::predict(s, Lq, p, prop, &ws, &diag);
    Eigen::VectorXd y = meas(pred.state.x);
    y(0) += 0.01 * std::sin(0.13 * static_cast<double>(k));
    y(2) -= 0.01 * std::cos(0.07 * static_cast<double>(k));

    s = bierman::filters::SRUKF::update(pred, y, Lr, meas, 0, &ws, &diag);

    if (!s.x.allFinite() || !s.L.allFinite()) {
      spdlog::error("Non-finite SRUKF values at step {}", k);
      return 1;
    }
  }

  // Fallbacks can occur, but they should not explode for this stress case.
  const int total_fallbacks = diag.predict_cov_fallbacks + diag.meas_cov_fallbacks + diag.state_cov_fallbacks;
  if (total_fallbacks > steps / 3) {
    spdlog::error("SRUKF fallback count too high: {}", total_fallbacks);
    return 1;
  }

  return 0;
}
