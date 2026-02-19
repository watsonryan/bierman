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
  constexpr int steps = 40;
  Eigen::MatrixXd A0 = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P0 = A0 * A0.transpose() + 1e-1 * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd x_truth = Eigen::VectorXd::Random(n);

  bierman::filters::KalmanState kf{Eigen::VectorXd::Zero(n), P0};
  bierman::filters::PotterState pot{Eigen::VectorXd::Zero(n), P0.llt().matrixL()};

  auto udf = bierman::linalg::ud_factorize(P0);
  bierman::filters::UDState ud{Eigen::VectorXd::Zero(n), udf.U, udf.d};

  Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
  Eigen::MatrixXd R0 = F0.llt().matrixU();
  bierman::filters::SRIFState s_hh{R0, Eigen::VectorXd::Zero(n)};
  bierman::filters::SRIFState s_qr{R0, Eigen::VectorXd::Zero(n)};

  const double sigma = 0.8;
  const double sw = 1.0 / sigma;
  const double r = sigma * sigma;

  for (int k = 0; k < steps; ++k) {
    Eigen::RowVectorXd H = Eigen::RowVectorXd::Random(n);
    const double z = H.dot(x_truth) + 0.05 * std::sin(0.17 * static_cast<double>(k));

    bierman::filters::KalmanJoseph::update_scalar(kf, H, z - H.dot(kf.x), sw);
    bierman::filters::PotterSR::update_scalar(pot, H, z - H.dot(pot.x), sw);
    bierman::filters::UDFilter::update_scalar(ud, H, z - H.dot(ud.x), r);

    Eigen::MatrixXd Hm(1, n);
    Hm.row(0) = H;
    Eigen::VectorXd zlin_hh(1);
    zlin_hh(0) = z;
    Eigen::VectorXd zlin_qr(1);
    zlin_qr(0) = z;
    Eigen::MatrixXd sqrtW(1, 1);
    sqrtW(0, 0) = sw;

    s_hh = bierman::filters::SRIF::update_householder(s_hh, Hm, zlin_hh, sqrtW).state;
    s_qr = bierman::filters::SRIF::update_qr_mgs(s_qr, Hm, zlin_qr, sqrtW).state;
  }

  const Eigen::VectorXd x_pot = pot.x;
  const Eigen::VectorXd x_ud = ud.x;
  const Eigen::VectorXd x_hh = bierman::linalg::solve_upper(s_hh.R, s_hh.z);
  const Eigen::VectorXd x_qr = bierman::linalg::solve_upper(s_qr.R, s_qr.z);

  const double e_pot = (kf.x - x_pot).norm();
  const double e_ud = (kf.x - x_ud).norm();
  const double e_hh = (kf.x - x_hh).norm();
  const double e_qr = (kf.x - x_qr).norm();

  if (e_pot > 5e-7 || e_ud > 5e-7 || e_hh > 5e-6 || e_qr > 1e-4) {
    spdlog::error("state mismatch pot={} ud={} hh={} qr={}", e_pot, e_ud, e_hh, e_qr);
    return 1;
  }

  Eigen::MatrixXd Ppot = pot.S * pot.S.transpose();
  Eigen::MatrixXd Pud = bierman::linalg::ud_reconstruct(ud.U, ud.d);
  Eigen::MatrixXd Fhh = s_hh.R.transpose() * s_hh.R;
  Eigen::MatrixXd Fqr = s_qr.R.transpose() * s_qr.R;
  Eigen::MatrixXd Phh = Fhh.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
  Eigen::MatrixXd Pqr = Fqr.ldlt().solve(Eigen::MatrixXd::Identity(n, n));

  const double epot = (Ppot - kf.P).norm() / kf.P.norm();
  const double eud = (Pud - kf.P).norm() / kf.P.norm();
  const double ehh = (Phh - kf.P).norm() / kf.P.norm();
  const double eqr = (Pqr - kf.P).norm() / kf.P.norm();

  if (epot > 1e-6 || eud > 1e-6 || ehh > 1e-5 || eqr > 1e-3) {
    spdlog::error("cov mismatch pot={} ud={} hh={} qr={}", epot, eud, ehh, eqr);
    return 1;
  }

  return 0;
}
