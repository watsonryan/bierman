#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::filters {

struct SchmidtState {
  Eigen::VectorXd x;
  Eigen::MatrixXd Px;
  Eigen::MatrixXd Py;
  Eigen::MatrixXd Pxy;
};

class SchmidtKalman {
 public:
  static void update(SchmidtState& s,
                     const Eigen::Ref<const Eigen::MatrixXd>& Ax,
                     const Eigen::Ref<const Eigen::MatrixXd>& Ay,
                     const Eigen::Ref<const Eigen::VectorXd>& delta,
                     const Eigen::Ref<const Eigen::MatrixXd>& R) {
    const Eigen::Index nx = s.x.size();
    if (s.Px.rows() != nx || s.Px.cols() != nx || s.Pxy.rows() != nx || s.Py.rows() != s.Py.cols()) {
      throw std::invalid_argument("Schmidt state dimension mismatch");
    }
    if (Ax.cols() != nx || Ay.cols() != s.Py.rows() || delta.size() != Ax.rows() || R.rows() != R.cols() || R.rows() != delta.size()) {
      throw std::invalid_argument("Schmidt update dimension mismatch");
    }

    const Eigen::MatrixXd S = Ax * s.Px * Ax.transpose() +
                              Ax * s.Pxy * Ay.transpose() +
                              Ay * s.Pxy.transpose() * Ax.transpose() +
                              Ay * s.Py * Ay.transpose() + R;

    const Eigen::MatrixXd K = (s.Px * Ax.transpose() + s.Pxy * Ay.transpose()) * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    s.x += K * delta;

    const Eigen::MatrixXd JF = Eigen::MatrixXd::Identity(nx, nx) - K * Ax;
    s.Px = JF * s.Px - K * Ay * s.Pxy.transpose();
    s.Pxy = JF * s.Pxy - K * Ay * s.Py;
    s.Px = 0.5 * (s.Px + s.Px.transpose());
  }
};

}  // namespace bierman::filters
