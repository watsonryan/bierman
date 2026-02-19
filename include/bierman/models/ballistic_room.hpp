#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::models {

class BallisticRoom {
 public:
  BallisticRoom(double drag_coeff, Eigen::Vector3d gravity)
      : drag_coeff_(drag_coeff), gravity_(gravity) {}

  Eigen::VectorXd propagate_truth(double dt, const Eigen::Ref<const Eigen::VectorXd>& x) const {
    if (x.size() != 6) {
      throw std::invalid_argument("BallisticRoom state must be size 6");
    }
    Eigen::VectorXd xn(6);
    const Eigen::Vector3d p = x.head<3>();
    const Eigen::Vector3d v = x.tail<3>();
    const Eigen::Vector3d a = gravity_ - drag_coeff_ * v;
    xn.head<3>() = p + dt * v;
    xn.tail<3>() = v + dt * a;
    return xn;
  }

  Eigen::VectorXd propagate_filter(double dt, const Eigen::Ref<const Eigen::VectorXd>& x) const {
    if (x.size() != 6) {
      throw std::invalid_argument("BallisticRoom state must be size 6");
    }
    Eigen::VectorXd xn(6);
    xn.head<3>() = x.head<3>() + dt * x.tail<3>();
    xn.tail<3>() = x.tail<3>();
    return xn;
  }

  Eigen::MatrixXd state_transition(double dt) const {
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(6, 6);
    Phi.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
    return Phi;
  }

  Eigen::MatrixXd process_noise_jacobian(double dt) const {
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(6, 3);
    G.block<3, 3>(0, 0) = 0.5 * dt * dt * Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 0) = dt * Eigen::Matrix3d::Identity();
    return G;
  }

 private:
  double drag_coeff_;
  Eigen::Vector3d gravity_;
};

}  // namespace bierman::models
