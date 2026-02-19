#pragma once

#include <stdexcept>

#include <Eigen/QR>

namespace bierman::linalg {

struct QR {
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
};

inline QR qr_householder(const Eigen::Ref<const Eigen::MatrixXd>& A) {
  const Eigen::Index m = A.rows();
  const Eigen::Index n = A.cols();
  if (m < n) {
    throw std::invalid_argument("qr_householder expects m >= n");
  }

  Eigen::HouseholderQR<Eigen::MatrixXd> decomp(A);
  Eigen::MatrixXd Q = decomp.householderQ() * Eigen::MatrixXd::Identity(m, n);
  Eigen::MatrixXd R = decomp.matrixQR().topRows(n).template triangularView<Eigen::Upper>();
  return {Q, R};
}

inline QR qr_mgs(const Eigen::Ref<const Eigen::MatrixXd>& A) {
  const Eigen::Index m = A.rows();
  const Eigen::Index n = A.cols();
  if (m < n) {
    throw std::invalid_argument("qr_mgs expects m >= n");
  }

  Eigen::MatrixXd Q = A;
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, n);

  for (Eigen::Index k = 0; k < n; ++k) {
    const double rkk = Q.col(k).norm();
    if (rkk <= 0.0) {
      throw std::runtime_error("qr_mgs encountered linearly dependent column");
    }
    R(k, k) = rkk;
    Q.col(k) /= rkk;

    for (Eigen::Index j = k + 1; j < n; ++j) {
      const double rkj = Q.col(k).dot(Q.col(j));
      R(k, j) = rkj;
      Q.col(j) -= Q.col(k) * rkj;
    }
  }

  return {Q, R};
}

}  // namespace bierman::linalg
