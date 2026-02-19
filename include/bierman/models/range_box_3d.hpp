#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::models {

class RangeOnlyBox3D {
 public:
  RangeOnlyBox3D(Eigen::MatrixXd tracker_pos, double sigma_range)
      : tracker_pos_(std::move(tracker_pos)), sigma_range_(sigma_range) {
    if (tracker_pos_.cols() != 3 || sigma_range_ <= 0.0) {
      throw std::invalid_argument("RangeOnlyBox3D invalid constructor arguments");
    }
  }

  Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x) const {
    if (x.size() < 3) {
      throw std::invalid_argument("RangeOnlyBox3D predict expects state with >= 3 elements");
    }
    Eigen::VectorXd y(tracker_pos_.rows());
    for (Eigen::Index i = 0; i < tracker_pos_.rows(); ++i) {
      const Eigen::Vector3d d = x.head<3>() - tracker_pos_.row(i).transpose();
      y(i) = d.norm();
    }
    return y;
  }

  Eigen::MatrixXd jacobian(const Eigen::Ref<const Eigen::VectorXd>& x) const {
    if (x.size() < 3) {
      throw std::invalid_argument("RangeOnlyBox3D jacobian expects state with >= 3 elements");
    }
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(tracker_pos_.rows(), x.size());
    for (Eigen::Index i = 0; i < tracker_pos_.rows(); ++i) {
      const Eigen::Vector3d d = x.head<3>() - tracker_pos_.row(i).transpose();
      const double r = d.norm();
      if (r > 0.0) {
        H.block<1, 3>(i, 0) = (d / r).transpose();
      }
    }
    return H;
  }

  Eigen::MatrixXd noise_cov() const {
    return sigma_range_ * sigma_range_ * Eigen::MatrixXd::Identity(tracker_pos_.rows(), tracker_pos_.rows());
  }

 private:
  Eigen::MatrixXd tracker_pos_;
  double sigma_range_;
};

}  // namespace bierman::models
