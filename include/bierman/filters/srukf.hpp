#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "bierman/filters/ukf.hpp"
#include "bierman/linalg/chol_update.hpp"
#include "bierman/linalg/triangular.hpp"

namespace bierman::filters {

struct SRUKFState {
  Eigen::VectorXd x;
  Eigen::MatrixXd L;  // Lower triangular covariance square-root
};

struct SRUKFPrediction {
  SRUKFState state;
  Eigen::MatrixXd sigma;
  UTWeights weights;
};

class SRUKF {
 private:
  static Eigen::MatrixXd robust_chol(const Eigen::Ref<const Eigen::MatrixXd>& Pin,
                                     const char* err_msg) {
    Eigen::MatrixXd P = 0.5 * (Pin + Pin.transpose());
    const Eigen::Index n = P.rows();
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    double jitter = 1e-12;
    for (int k = 0; k < 12; ++k) {
      Eigen::LLT<Eigen::MatrixXd> llt(P + jitter * I);
      if (llt.info() == Eigen::Success) {
        return llt.matrixL();
      }
      jitter *= 10.0;
    }
    throw std::runtime_error(err_msg);
  }

 public:
  template <typename PropagateFn>
  static SRUKFPrediction predict(const SRUKFState& in,
                                 const Eigen::Ref<const Eigen::MatrixXd>& Lq,
                                 const UTParams& p,
                                 PropagateFn&& propagate) {
    const Eigen::Index n = in.x.size();
    if (in.L.rows() != n || in.L.cols() != n || Lq.rows() != n || Lq.cols() != n) {
      throw std::invalid_argument("SRUKF predict dimension mismatch");
    }

    SRUKFPrediction pred;
    pred.weights = sigma_weights(n, p);

    const double c = static_cast<double>(n) + pred.weights.lambda;
    if (c <= 0.0) {
      throw std::invalid_argument("SRUKF invalid UT scaling");
    }

    Eigen::MatrixXd Chi(n, 2 * n + 1);
    Chi.col(0) = in.x;
    const double scale = std::sqrt(c);
    for (Eigen::Index i = 0; i < n; ++i) {
      const Eigen::VectorXd v = scale * in.L.col(i);
      Chi.col(1 + i) = in.x + v;
      Chi.col(1 + n + i) = in.x - v;
    }

    pred.sigma.resize(n, Chi.cols());
    for (Eigen::Index i = 0; i < Chi.cols(); ++i) {
      pred.sigma.col(i) = propagate(Chi.col(i));
    }

    pred.state.x = Eigen::VectorXd::Zero(n);
    for (Eigen::Index i = 0; i < pred.sigma.cols(); ++i) {
      pred.state.x += pred.weights.wm(i) * pred.sigma.col(i);
    }

    pred.state.L = Lq;
    bool fallback = false;
    for (Eigen::Index i = 0; i < pred.sigma.cols(); ++i) {
      const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
      const double w = pred.weights.wc(i);
      if (w == 0.0) {
        continue;
      }
      const double s = std::sqrt(std::abs(w));
      const bool ok = bierman::linalg::chol_rank1_update_lower(pred.state.L, s * dx, w > 0.0 ? 1.0 : -1.0);
      if (!ok) {
        fallback = true;
        break;
      }
    }

    if (fallback) {
      Eigen::MatrixXd P = Lq * Lq.transpose();
      for (Eigen::Index i = 0; i < pred.sigma.cols(); ++i) {
        const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
        P += pred.weights.wc(i) * (dx * dx.transpose());
      }
      pred.state.L = robust_chol(P, "SRUKF predict covariance factorization failed");
    }

    return pred;
  }

  template <typename MeasureFn>
  static SRUKFState update(const SRUKFPrediction& pred,
                           const Eigen::Ref<const Eigen::VectorXd>& y,
                           const Eigen::Ref<const Eigen::MatrixXd>& Lr,
                           MeasureFn&& measure,
                           Eigen::Index consider_count = 0) {
    const Eigen::Index n = pred.state.x.size();
    const Eigen::Index nsig = pred.sigma.cols();

    Eigen::VectorXd y0 = measure(pred.sigma.col(0));
    const Eigen::Index m = y0.size();
    if (y.size() != m || Lr.rows() != m || Lr.cols() != m) {
      throw std::invalid_argument("SRUKF update dimension mismatch");
    }

    Eigen::MatrixXd Y(m, nsig);
    Y.col(0) = y0;
    for (Eigen::Index i = 1; i < nsig; ++i) {
      Y.col(i) = measure(pred.sigma.col(i));
    }

    Eigen::VectorXd ybar = Eigen::VectorXd::Zero(m);
    for (Eigen::Index i = 0; i < nsig; ++i) {
      ybar += pred.weights.wm(i) * Y.col(i);
    }

    Eigen::MatrixXd Sy = Lr;
    Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(n, m);
    bool sy_fallback = false;

    for (Eigen::Index i = 0; i < nsig; ++i) {
      const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
      const Eigen::VectorXd dy = Y.col(i) - ybar;
      const double w = pred.weights.wc(i);
      if (w != 0.0) {
        const bool ok = bierman::linalg::chol_rank1_update_lower(Sy, std::sqrt(std::abs(w)) * dy, w > 0.0 ? 1.0 : -1.0);
        if (!ok) {
          sy_fallback = true;
        }
      }
      Pxy += w * dx * dy.transpose();
    }

    if (sy_fallback) {
      Eigen::MatrixXd Pyy = Lr * Lr.transpose();
      for (Eigen::Index i = 0; i < nsig; ++i) {
        const Eigen::VectorXd dy = Y.col(i) - ybar;
        Pyy += pred.weights.wc(i) * (dy * dy.transpose());
      }
      Sy = robust_chol(Pyy, "SRUKF measurement covariance factorization failed");
    }

    Eigen::MatrixXd Sy_inv_PxyT = Sy.triangularView<Eigen::Lower>().solve(Pxy.transpose());
    Eigen::MatrixXd Kt = Sy.transpose().triangularView<Eigen::Upper>().solve(Sy_inv_PxyT);
    Eigen::MatrixXd K = Kt.transpose();

    SRUKFState out = pred.state;
    if (consider_count > 0) {
      if (consider_count > n) {
        throw std::invalid_argument("consider_count exceeds state dimension");
      }
      K.bottomRows(consider_count).setZero();
    }

    out.x += K * (y - ybar);

    const Eigen::MatrixXd L_before = out.L;
    const Eigen::MatrixXd U = K * Sy.transpose();

    bool state_fallback = false;
    for (Eigen::Index i = 0; i < U.cols(); ++i) {
      const bool ok = bierman::linalg::chol_rank1_update_lower(out.L, U.col(i), -1.0);
      if (!ok) {
        state_fallback = true;
        break;
      }
    }

    if (state_fallback) {
      Eigen::MatrixXd Ppred = L_before * L_before.transpose();
      Eigen::MatrixXd Pyy = Sy * Sy.transpose();
      Eigen::MatrixXd P = Ppred - K * Pyy * K.transpose();
      out.L = robust_chol(P, "SRUKF state covariance factorization failed");
    }

    if (consider_count > 0) {
      const Eigen::Index ns = n - consider_count;
      out.L.block(ns, 0, consider_count, n) = L_before.block(ns, 0, consider_count, n);
    }

    return out;
  }
};

}  // namespace bierman::filters
