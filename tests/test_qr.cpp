#include <cmath>

#include <spdlog/spdlog.h>

#include "bierman/linalg/qr.hpp"

int main() {
  constexpr int m = 12;
  constexpr int n = 6;
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);

  const auto mgs = bierman::linalg::qr_mgs(A);
  const auto hh = bierman::linalg::qr_householder(A);

  const double rec_mgs = (A - mgs.Q * mgs.R).norm() / A.norm();
  const double rec_hh = (A - hh.Q * hh.R).norm() / A.norm();

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  const double ortho_mgs = (mgs.Q.transpose() * mgs.Q - I).norm();
  const double ortho_hh = (hh.Q.transpose() * hh.Q - I).norm();

  if (!std::isfinite(rec_mgs) || rec_mgs > 1e-10) {
    spdlog::error("MGS QR reconstruction error too high: {}", rec_mgs);
    return 1;
  }
  if (!std::isfinite(rec_hh) || rec_hh > 1e-10) {
    spdlog::error("Householder QR reconstruction error too high: {}", rec_hh);
    return 1;
  }
  if (!std::isfinite(ortho_mgs) || ortho_mgs > 1e-10) {
    spdlog::error("MGS Q orthogonality error too high: {}", ortho_mgs);
    return 1;
  }
  if (!std::isfinite(ortho_hh) || ortho_hh > 1e-10) {
    spdlog::error("Householder Q orthogonality error too high: {}", ortho_hh);
    return 1;
  }

  return 0;
}
