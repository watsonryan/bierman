#pragma once

#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/Core>

namespace bierman::filters {

struct UTParams {
  double alpha = 1e-3;
  double kappa = 0.0;
  double beta = 2.0;
};

struct UTWeights {
  Eigen::VectorXd wm;
  Eigen::VectorXd wc;
  double lambda = 0.0;
};

struct UKFState {
  Eigen::VectorXd x;
  Eigen::MatrixXd P;
};

struct UKFPrediction {
  UKFState state;
  Eigen::MatrixXd sigma;  // Propagated sigma points
  UTWeights weights;
};

// Workspace to reduce repeated allocations in tight filter loops.
struct UKFWorkspace {
  Eigen::MatrixXd Chi;
  Eigen::MatrixXd Y;
  Eigen::VectorXd ybar;
  Eigen::MatrixXd Pyy;
  Eigen::MatrixXd Pxy;
};

inline UTWeights sigma_weights(Eigen::Index n, const UTParams& p) {
  const double lambda = p.alpha * p.alpha * (static_cast<double>(n) + p.kappa) - static_cast<double>(n);
  const double c = static_cast<double>(n) + lambda;
  if (c <= 0.0) {
    throw std::invalid_argument("UT scaling is non-positive");
  }

  UTWeights w;
  w.lambda = lambda;
  w.wm = Eigen::VectorXd::Constant(2 * n + 1, 1.0 / (2.0 * c));
  w.wc = w.wm;
  w.wm(0) = lambda / c;
  w.wc(0) = lambda / c + (1.0 - p.alpha * p.alpha + p.beta);
  return w;
}

inline Eigen::MatrixXd sigma_points(const Eigen::Ref<const Eigen::VectorXd>& x,
                                    const Eigen::Ref<const Eigen::MatrixXd>& P,
                                    const UTParams& p) {
  if (P.rows() != P.cols() || P.rows() != x.size()) {
    throw std::invalid_argument("sigma_points dimension mismatch");
  }

  const Eigen::Index n = x.size();
  const auto w = sigma_weights(n, p);
  const double c = static_cast<double>(n) + w.lambda;

  Eigen::LLT<Eigen::MatrixXd> llt(P);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("sigma_points requires SPD covariance");
  }
  const Eigen::MatrixXd L = llt.matrixL();

  Eigen::MatrixXd Chi(n, 2 * n + 1);
  Chi.col(0) = x;
  const double scale = std::sqrt(c);
  for (Eigen::Index i = 0; i < n; ++i) {
    const Eigen::VectorXd v = scale * L.col(i);
    Chi.col(1 + i) = x + v;
    Chi.col(1 + n + i) = x - v;
  }
  return Chi;
}

template <typename PropagateFn>
inline UKFPrediction predict(const UKFState& in,
                             const Eigen::Ref<const Eigen::MatrixXd>& Q,
                             const UTParams& p,
                             PropagateFn&& propagate,
                             UKFWorkspace* ws = nullptr) {
  const Eigen::Index n = in.x.size();
  if (in.P.rows() != n || in.P.cols() != n || Q.rows() != n || Q.cols() != n) {
    throw std::invalid_argument("UKF predict dimension mismatch");
  }

  UKFPrediction pred;
  pred.weights = sigma_weights(n, p);

  const Eigen::MatrixXd Chi_tmp = sigma_points(in.x, in.P, p);
  const Eigen::MatrixXd* Chi_ptr = &Chi_tmp;
  if (ws != nullptr) {
    ws->Chi = Chi_tmp;
    Chi_ptr = &ws->Chi;
  }

  pred.sigma.resize(n, Chi_ptr->cols());
  for (Eigen::Index i = 0; i < Chi_ptr->cols(); ++i) {
    pred.sigma.col(i) = propagate(Chi_ptr->col(i));
  }

  pred.state.x = Eigen::VectorXd::Zero(n);
  for (Eigen::Index i = 0; i < pred.sigma.cols(); ++i) {
    pred.state.x += pred.weights.wm(i) * pred.sigma.col(i);
  }

  pred.state.P = Q;
  for (Eigen::Index i = 0; i < pred.sigma.cols(); ++i) {
    const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
    pred.state.P += pred.weights.wc(i) * (dx * dx.transpose());
  }
  pred.state.P = 0.5 * (pred.state.P + pred.state.P.transpose());

  return pred;
}

template <typename MeasureFn>
inline UKFState update(const UKFPrediction& pred,
                       const Eigen::Ref<const Eigen::VectorXd>& y,
                       const Eigen::Ref<const Eigen::MatrixXd>& R,
                       MeasureFn&& measure,
                       UKFWorkspace* ws = nullptr) {
  const Eigen::Index n = pred.state.x.size();
  const Eigen::Index nsig = pred.sigma.cols();

  Eigen::VectorXd y0 = measure(pred.sigma.col(0));
  const Eigen::Index m = y0.size();
  if (y.size() != m || R.rows() != m || R.cols() != m) {
    throw std::invalid_argument("UKF update dimension mismatch");
  }

  Eigen::MatrixXd Y_local(m, nsig);
  Eigen::MatrixXd* Y = &Y_local;
  if (ws != nullptr) {
    ws->Y.resize(m, nsig);
    Y = &ws->Y;
  }
  Y->col(0) = y0;
  for (Eigen::Index i = 1; i < nsig; ++i) {
    Y->col(i) = measure(pred.sigma.col(i));
  }

  Eigen::VectorXd ybar_local = Eigen::VectorXd::Zero(m);
  Eigen::VectorXd* ybar = &ybar_local;
  if (ws != nullptr) {
    ws->ybar = Eigen::VectorXd::Zero(m);
    ybar = &ws->ybar;
  }

  for (Eigen::Index i = 0; i < nsig; ++i) {
    *ybar += pred.weights.wm(i) * Y->col(i);
  }

  Eigen::MatrixXd Pyy_local = R;
  Eigen::MatrixXd Pxy_local = Eigen::MatrixXd::Zero(n, m);
  Eigen::MatrixXd* Pyy = &Pyy_local;
  Eigen::MatrixXd* Pxy = &Pxy_local;
  if (ws != nullptr) {
    ws->Pyy = R;
    ws->Pxy = Eigen::MatrixXd::Zero(n, m);
    Pyy = &ws->Pyy;
    Pxy = &ws->Pxy;
  }

  for (Eigen::Index i = 0; i < nsig; ++i) {
    const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
    const Eigen::VectorXd dy = Y->col(i) - *ybar;
    *Pyy += pred.weights.wc(i) * (dy * dy.transpose());
    *Pxy += pred.weights.wc(i) * (dx * dy.transpose());
  }

  const Eigen::MatrixXd K = (*Pxy) * Pyy->ldlt().solve(Eigen::MatrixXd::Identity(m, m));

  UKFState out = pred.state;
  out.x += K * (y - *ybar);
  out.P -= K * (*Pyy) * K.transpose();
  out.P = 0.5 * (out.P + out.P.transpose());
  return out;
}

}  // namespace bierman::filters
