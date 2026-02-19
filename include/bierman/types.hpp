#pragma once

#include <Eigen/Core>

namespace bierman {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;

struct Stats {
  long long rhs_evals = 0;
  int steps = 0;
};

}  // namespace bierman
