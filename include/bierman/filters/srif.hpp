#pragma once

#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/QR>

#include "bierman/linalg/qr.hpp"
#include "bierman/linalg/triangular.hpp"

namespace bierman::filters {

struct SRIFState {
  Eigen::MatrixXd R;  // Upper triangular square-root information
  Eigen::VectorXd z;  // RHS in R*x = z
};

class SRIF {
 public:
  struct UpdateResult {
    SRIFState state;
    Eigen::VectorXd x;
    double residual_ss = 0.0;
  };

  // Convention: pass linearized pseudo-observations in absolute-state form,
  // z_lin = y - h(x_ref) + H*x_ref, so the information solve returns absolute x.
  static UpdateResult update_householder(const SRIFState& prior,
                                         const Eigen::Ref<const Eigen::MatrixXd>& H,
                                         const Eigen::Ref<const Eigen::VectorXd>& z_lin,
                                         const Eigen::Ref<const Eigen::MatrixXd>& sqrtW) {
    const Eigen::Index n = prior.R.rows();
    if (prior.R.rows() != prior.R.cols() || prior.z.size() != n) {
      throw std::invalid_argument("SRIF prior dimension mismatch");
    }
    if (H.cols() != n || H.rows() != z_lin.size() || sqrtW.rows() != z_lin.size() || sqrtW.cols() != z_lin.size()) {
      throw std::invalid_argument("SRIF update dimension mismatch");
    }

    Eigen::MatrixXd A(prior.R.rows() + H.rows(), n + 1);
    A.topRows(n).leftCols(n) = prior.R;
    A.topRows(n).col(n) = prior.z;
    A.bottomRows(H.rows()).leftCols(n) = sqrtW * H;
    A.bottomRows(H.rows()).col(n) = sqrtW * z_lin;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd T = qr.householderQ().adjoint() * A;

    SRIFState out;
    out.R = T.topRows(n).leftCols(n).template triangularView<Eigen::Upper>();
    out.z = T.topRows(n).col(n);

    UpdateResult res;
    res.state = out;
    res.x = bierman::linalg::solve_upper(out.R, out.z);
    if (T.rows() > n) {
      res.residual_ss = T.bottomRows(T.rows() - n).col(n).squaredNorm();
    }
    return res;
  }

  // Same linearization convention as update_householder().
  static UpdateResult update_qr_mgs(const SRIFState& prior,
                                    const Eigen::Ref<const Eigen::MatrixXd>& H,
                                    const Eigen::Ref<const Eigen::VectorXd>& z_lin,
                                    const Eigen::Ref<const Eigen::MatrixXd>& sqrtW) {
    const Eigen::Index n = prior.R.rows();
    if (prior.R.rows() != prior.R.cols() || prior.z.size() != n) {
      throw std::invalid_argument("SRIF prior dimension mismatch");
    }

    Eigen::MatrixXd A(n + H.rows(), n);
    A.topRows(n) = prior.R;
    A.bottomRows(H.rows()) = sqrtW * H;

    Eigen::VectorXd y(n + z_lin.size());
    y.head(n) = prior.z;
    y.tail(z_lin.size()) = sqrtW * z_lin;

    const auto qr = bierman::linalg::qr_mgs(A);
    const Eigen::VectorXd qty = qr.Q.transpose() * y;

    UpdateResult res;
    res.state.R = qr.R;
    res.state.z = qty.head(n);
    res.x = bierman::linalg::solve_upper(res.state.R, res.state.z);
    if (qty.size() > n) {
      res.residual_ss = qty.tail(qty.size() - n).squaredNorm();
    }
    return res;
  }

  static SRIFState predict(const SRIFState& prior,
                           const Eigen::Ref<const Eigen::MatrixXd>& PhiInv,
                           const Eigen::Ref<const Eigen::MatrixXd>& Rw,
                           const Eigen::Ref<const Eigen::MatrixXd>& G) {
    const Eigen::Index n = prior.R.rows();
    const Eigen::Index m = Rw.rows();

    if (prior.R.rows() != prior.R.cols() || prior.z.size() != n) {
      throw std::invalid_argument("SRIF prior dimension mismatch");
    }
    if (PhiInv.rows() != n || PhiInv.cols() != n || G.rows() != n || G.cols() != m || Rw.cols() != m) {
      throw std::invalid_argument("SRIF predict dimension mismatch");
    }

    Eigen::MatrixXd Rprop = prior.R * PhiInv;

    Eigen::MatrixXd A(m + n, m + n + 1);
    A.setZero();
    A.topLeftCorner(m, m) = Rw;
    A.bottomLeftCorner(n, m) = -Rprop * G;
    A.block(m, m, n, n) = Rprop;
    A.block(m, m + n, n, 1) = prior.z;

    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd T = qr.householderQ().adjoint() * A;

    SRIFState out;
    out.R = T.block(m, m, n, n).template triangularView<Eigen::Upper>();
    out.z = T.block(m, m + n, n, 1);
    return out;
  }
};

}  // namespace bierman::filters
