#pragma once

#include <Eigen/Core>

namespace bierman::linalg {

inline Eigen::VectorXd solve_upper(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                   const Eigen::Ref<const Eigen::VectorXd>& b) {
  return R.triangularView<Eigen::Upper>().solve(b);
}

inline Eigen::VectorXd solve_lower(const Eigen::Ref<const Eigen::MatrixXd>& L,
                                   const Eigen::Ref<const Eigen::VectorXd>& b) {
  return L.triangularView<Eigen::Lower>().solve(b);
}

inline void solve_upper_in_place(const Eigen::Ref<const Eigen::MatrixXd>& R,
                                 Eigen::Ref<Eigen::VectorXd> b) {
  b = R.triangularView<Eigen::Upper>().solve(b);
}

inline void solve_lower_in_place(const Eigen::Ref<const Eigen::MatrixXd>& L,
                                 Eigen::Ref<Eigen::VectorXd> b) {
  b = L.triangularView<Eigen::Lower>().solve(b);
}

}  // namespace bierman::linalg
