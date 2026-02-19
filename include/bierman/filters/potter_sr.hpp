#pragma once

#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace bierman::filters {

struct PotterState {
  Eigen::VectorXd x;
  Eigen::MatrixXd S;  // Lower triangular, P = S*S^T
};

class PotterSR {
 public:
  static void update_scalar(PotterState& s,
                            const Eigen::Ref<const Eigen::RowVectorXd>& A,
                            double delta,
                            double sw) {
    if (s.S.rows() != s.S.cols() || s.S.rows() != s.x.size()) {
      throw std::invalid_argument("PotterSR state dimension mismatch");
    }
    if (A.size() != s.x.size()) {
      throw std::invalid_argument("PotterSR measurement jacobian dimension mismatch");
    }
    if (sw <= 0.0) {
      throw std::invalid_argument("PotterSR sw must be positive");
    }

    const double delta_w = sw * delta;
    const Eigen::RowVectorXd A_w = sw * A;

    const Eigen::RowVectorXd vtrans = A_w * s.S;
    const double sigma = 1.0 / (vtrans.squaredNorm() + 1.0);
    const Eigen::VectorXd K = s.S * vtrans.transpose();
    s.x += K * (delta_w * sigma);

    const double lambda = sigma / (1.0 + std::sqrt(sigma));
    Eigen::MatrixXd Snew = s.S - (lambda * K) * vtrans;

    // Force lower-triangular square-root form for downstream consistency.
    Eigen::MatrixXd Pnew = Snew * Snew.transpose();
    Pnew = 0.5 * (Pnew + Pnew.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(Pnew);
    if (llt.info() != Eigen::Success) {
      throw std::runtime_error("PotterSR produced non-SPD covariance");
    }
    s.S = llt.matrixL();
  }
};

}  // namespace bierman::filters
