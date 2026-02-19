#pragma once

#include <cmath>

#include <Eigen/Core>

namespace bierman::util {

inline double rms(const Eigen::Ref<const Eigen::VectorXd>& x) {
  if (x.size() == 0) {
    return 0.0;
  }
  return std::sqrt(x.squaredNorm() / static_cast<double>(x.size()));
}

inline double nees(const Eigen::Ref<const Eigen::VectorXd>& e,
                   const Eigen::Ref<const Eigen::MatrixXd>& P) {
  return (e.transpose() * P.ldlt().solve(e))(0);
}

}  // namespace bierman::util
