#pragma once

#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Cholesky>

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

    s.P = IKH * s.P * IKH.transpose() + K * K.transpose();
    s.P = 0.5 * (s.P + s.P.transpose());
  }

  template <typename PredictFn, typename JacobianFn>
  static void update_scalar_iterated(KalmanState& s,
                                     double y,
                                     double sw,
                                     PredictFn&& predict,
                                     JacobianFn&& jacobian,
                                     int max_iters = 4,
                                     double tol = 1e-8) {
    KalmanState prior = s;
    Eigen::VectorXd x_ref = s.x;

    KalmanState best = s;
    double best_cost = std::numeric_limits<double>::infinity();

    for (int it = 0; it < max_iters; ++it) {
      KalmanState tmp = prior;
      const double yhat = predict(x_ref);
      const Eigen::RowVectorXd H = jacobian(x_ref);
      update_scalar(tmp, H, y - yhat, sw);

      const Eigen::VectorXd dx = tmp.x - prior.x;
      const double prior_cost = (dx.transpose() * prior.P.ldlt().solve(dx))(0);
      const double meas_res = y - predict(tmp.x);
      const double meas_cost = (sw * meas_res) * (sw * meas_res);
      const double cost = prior_cost + meas_cost;

      if (cost < best_cost) {
        best_cost = cost;
        best = tmp;
      }

      if ((tmp.x - x_ref).norm() < tol) {
        break;
      }
      x_ref = tmp.x;
    }

    s = best;
  }
};

}  // namespace bierman::filters
