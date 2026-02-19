#pragma once

#include <stdexcept>

#include <Eigen/Core>

namespace bierman::filters {

struct UDState {
  Eigen::VectorXd x;
  Eigen::MatrixXd U;  // Unit upper
  Eigen::VectorXd d;  // Diagonal of D
};

class UDFilter {
 public:
  static void update_scalar(UDState& s,
                            const Eigen::Ref<const Eigen::RowVectorXd>& a,
                            double delta,
                            double r) {
    if (s.U.rows() != s.U.cols() || s.U.rows() != s.x.size() || s.d.size() != s.x.size()) {
      throw std::invalid_argument("UDFilter state dimension mismatch");
    }
    if (a.size() != s.x.size()) {
      throw std::invalid_argument("UDFilter measurement jacobian dimension mismatch");
    }
    if (r <= 0.0) {
      throw std::invalid_argument("UDFilter r must be positive");
    }

    const Eigen::Index n = s.x.size();
    Eigen::VectorXd av = s.U.transpose() * a.transpose();
    Eigen::VectorXd b = s.d.asDiagonal() * av;

    double alpha = r + b(0) * av(0);
    double gamma = 1.0 / alpha;
    s.d(0) = r * gamma * s.d(0);

    for (Eigen::Index j = 1; j < n; ++j) {
      double beta = alpha;
      alpha += b(j) * av(j);
      double lambda = -av(j) * gamma;
      gamma = 1.0 / alpha;
      s.d(j) = beta * gamma * s.d(j);

      for (Eigen::Index i = 0; i < j; ++i) {
        const double uij = s.U(i, j);
        s.U(i, j) = uij + b(i) * lambda;
        b(i) = b(i) + b(j) * uij;
      }
    }

    s.x += b * (delta * gamma);
  }

  static void predict_diagonal_q(UDState& s,
                                 const Eigen::Ref<const Eigen::MatrixXd>& Phi,
                                 const Eigen::Ref<const Eigen::VectorXd>& q_diag,
                                 const Eigen::Ref<const Eigen::MatrixXd>& G) {
    if (Phi.rows() != Phi.cols() || Phi.rows() != s.x.size()) {
      throw std::invalid_argument("UDFilter Phi dimension mismatch");
    }
    if (G.rows() != s.x.size() || G.cols() != q_diag.size()) {
      throw std::invalid_argument("UDFilter G/Q dimension mismatch");
    }

    s.x = Phi * s.x;

    const Eigen::Index n = s.x.size();
    const Eigen::Index m = q_diag.size();
    const Eigen::Index N = n + m;

    Eigen::MatrixXd A(N, n);
    A.topRows(n) = Phi * s.U;
    A.bottomRows(m) = G.transpose();

    Eigen::VectorXd dtmp(N);
    dtmp.head(n) = s.d;
    dtmp.tail(m) = q_diag;

    for (Eigen::Index ii = n - 1; ii >= 0; --ii) {
      const Eigen::VectorXd c = dtmp.asDiagonal() * A.col(ii);
      s.d(ii) = A.col(ii).dot(c);
      const Eigen::VectorXd dvec = c / s.d(ii);
      for (Eigen::Index jj = 0; jj < ii; ++jj) {
        s.U(jj, ii) = A.col(jj).dot(dvec);
        A.col(jj) = A.col(jj) - s.U(jj, ii) * A.col(ii);
      }
    }

    s.U.diagonal().setOnes();
    for (Eigen::Index r = 0; r < n; ++r) {
      for (Eigen::Index c = 0; c < r; ++c) {
        s.U(r, c) = 0.0;
      }
    }
  }
};

}  // namespace bierman::filters
