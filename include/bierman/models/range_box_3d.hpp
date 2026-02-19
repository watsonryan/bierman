#pragma once

#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/SVD>

namespace bierman::models {

struct ObservabilityReport {
  int rank = 0;
  double condition = std::numeric_limits<double>::infinity();
  double sigma_min = 0.0;
  double sigma_max = 0.0;

  bool well_conditioned(int required_rank = 3, double max_condition = 1e8) const {
    return rank >= required_rank && std::isfinite(condition) && condition <= max_condition;
  }
};

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

  ObservabilityReport observability(const Eigen::Ref<const Eigen::VectorXd>& x,
                                    double svd_tol = 1e-10) const {
    Eigen::MatrixXd H = jacobian(x);
    H = H.leftCols(3);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd s = svd.singularValues();

    ObservabilityReport rep;
    rep.sigma_max = s.size() > 0 ? s(0) : 0.0;
    rep.sigma_min = s.size() > 0 ? s(s.size() - 1) : 0.0;

    const double thresh = svd_tol * (rep.sigma_max > 0.0 ? rep.sigma_max : 1.0);
    rep.rank = 0;
    for (Eigen::Index i = 0; i < s.size(); ++i) {
      if (s(i) > thresh) {
        ++rep.rank;
      }
    }

    if (rep.sigma_min > 0.0) {
      rep.condition = rep.sigma_max / rep.sigma_min;
    }
    return rep;
  }

 private:
  Eigen::MatrixXd tracker_pos_;
  double sigma_range_;
};

}  // namespace bierman::models
