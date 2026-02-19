#pragma once

#include <cmath>
#include <functional>
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
                             PropagateFn&& propagate) {
  const Eigen::Index n = in.x.size();
  if (in.P.rows() != n || in.P.cols() != n || Q.rows() != n || Q.cols() != n) {
    throw std::invalid_argument("UKF predict dimension mismatch");
  }

  UKFPrediction pred;
  pred.weights = sigma_weights(n, p);

  const Eigen::MatrixXd Chi = sigma_points(in.x, in.P, p);
  pred.sigma.resize(n, Chi.cols());
  for (Eigen::Index i = 0; i < Chi.cols(); ++i) {
    pred.sigma.col(i) = propagate(Chi.col(i));
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
                       MeasureFn&& measure) {
  const Eigen::Index n = pred.state.x.size();
  const Eigen::Index nsig = pred.sigma.cols();

  Eigen::VectorXd y0 = measure(pred.sigma.col(0));
  const Eigen::Index m = y0.size();
  if (y.size() != m || R.rows() != m || R.cols() != m) {
    throw std::invalid_argument("UKF update dimension mismatch");
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

  Eigen::MatrixXd Pyy = R;
  Eigen::MatrixXd Pxy = Eigen::MatrixXd::Zero(n, m);
  for (Eigen::Index i = 0; i < nsig; ++i) {
    const Eigen::VectorXd dx = pred.sigma.col(i) - pred.state.x;
    const Eigen::VectorXd dy = Y.col(i) - ybar;
    Pyy += pred.weights.wc(i) * (dy * dy.transpose());
    Pxy += pred.weights.wc(i) * (dx * dy.transpose());
  }

  const Eigen::MatrixXd K = Pxy * Pyy.ldlt().solve(Eigen::MatrixXd::Identity(m, m));

  UKFState out = pred.state;
  out.x += K * (y - ybar);
  out.P -= K * Pyy * K.transpose();
  out.P = 0.5 * (out.P + out.P.transpose());
  return out;
}

}  // namespace bierman::filters
