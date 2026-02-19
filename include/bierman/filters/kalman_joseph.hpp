#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::filters {

struct KalmanState {
  Eigen::VectorXd x;
  Eigen::MatrixXd P;
};

class KalmanJoseph {
 public:
  static void update_scalar(KalmanState& s,
                            const Eigen::Ref<const Eigen::RowVectorXd>& A,
                            double delta,
                            double sw) {
    if (s.P.rows() != s.P.cols() || s.P.rows() != s.x.size()) {
      throw std::invalid_argument("KalmanJoseph state dimension mismatch");
    }
    if (A.size() != s.x.size()) {
      throw std::invalid_argument("KalmanJoseph measurement jacobian dimension mismatch");
    }
    if (sw <= 0.0) {
      throw std::invalid_argument("KalmanJoseph sw must be positive");
    }

    const Eigen::Index n = s.x.size();

    const double delta_w = sw * delta;
    const Eigen::RowVectorXd A_w = sw * A;

    const Eigen::VectorXd v = s.P * A_w.transpose();
    const double sigma = (A_w * v)(0) + 1.0;
    const Eigen::VectorXd K = v / sigma;

    s.x += K * delta_w;

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    const Eigen::MatrixXd IKH = I - K * A_w;

    // Joseph-stabilized covariance update with whitened measurement variance R=1.
    s.P = IKH * s.P * IKH.transpose() + K * K.transpose();
    s.P = 0.5 * (s.P + s.P.transpose());
  }
};

}  // namespace bierman::filters
