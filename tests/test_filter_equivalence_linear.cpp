#include <cmath>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/potter_sr.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/linalg/triangular.hpp"
#include "bierman/linalg/ud.hpp"

int main() {
  constexpr int n = 7;
  Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P0 = A0 * A0.transpose() + 1e-1 * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Eigen::RowVectorXd H = Eigen::RowVectorXd::Random(n);

  const double sigma = 0.8;
  const double sw = 1.0 / sigma;
  const double r = sigma * sigma;
  const double z = 0.13;

  bierman::filters::KalmanState kf{x0, P0};
  bierman::filters::KalmanJoseph::update_scalar(kf, H, z - H.dot(x0), sw);

  bierman::filters::PotterState pot{x0, P0.llt().matrixL()};
  bierman::filters::PotterSR::update_scalar(pot, H, z - H.dot(x0), sw);

  auto udf = bierman::linalg::ud_factorize(P0);
  bierman::filters::UDState ud{x0, udf.U, udf.d};
  bierman::filters::UDFilter::update_scalar(ud, H, z - H.dot(x0), r);

  Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
  Eigen::MatrixXd R0 = F0.llt().matrixU();
  bierman::filters::SRIFState s0{R0, R0 * x0};
  Eigen::MatrixXd Hm(1, n);
  Hm.row(0) = H;
  Eigen::VectorXd zm(1);
  zm(0) = z;
  Eigen::MatrixXd sqrtW(1, 1);
  sqrtW(0, 0) = sw;

  auto hh = bierman::filters::SRIF::update_householder(s0, Hm, zm, sqrtW);
  auto qr = bierman::filters::SRIF::update_qr_mgs(s0, Hm, zm, sqrtW);

  const Eigen::VectorXd x_pot = pot.x;
  const Eigen::VectorXd x_ud = ud.x;
  const Eigen::VectorXd x_hh = bierman::linalg::solve_upper(hh.state.R, hh.state.z);
  const Eigen::VectorXd x_qr = bierman::linalg::solve_upper(qr.state.R, qr.state.z);

  const double e_pot = (kf.x - x_pot).norm();
  const double e_ud = (kf.x - x_ud).norm();
  const double e_hh = (kf.x - x_hh).norm();
  const double e_qr = (kf.x - x_qr).norm();

  if (e_pot > 1e-8 || e_ud > 1e-8 || e_hh > 1e-7 || e_qr > 1e-6) {
    spdlog::error("state mismatch pot={} ud={} hh={} qr={}", e_pot, e_ud, e_hh, e_qr);
    return 1;
  }

  Eigen::MatrixXd Ppot = pot.S * pot.S.transpose();
  Eigen::MatrixXd Pud = bierman::linalg::ud_reconstruct(ud.U, ud.d);
  Eigen::MatrixXd Finf = hh.state.R.transpose() * hh.state.R;
  Eigen::MatrixXd Phh = Finf.ldlt().solve(Eigen::MatrixXd::Identity(n, n));

  const double epot = (Ppot - kf.P).norm() / kf.P.norm();
  const double eud = (Pud - kf.P).norm() / kf.P.norm();
  const double ehh = (Phh - kf.P).norm() / kf.P.norm();

  if (epot > 1e-8 || eud > 1e-8 || ehh > 1e-6) {
    spdlog::error("cov mismatch pot={} ud={} hh={}", epot, eud, ehh);
    return 1;
  }

  return 0;
}
