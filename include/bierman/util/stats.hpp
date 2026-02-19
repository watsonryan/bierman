#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Cholesky>

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

inline double nis(const Eigen::Ref<const Eigen::VectorXd>& innov,
                  const Eigen::Ref<const Eigen::MatrixXd>& S) {
  return (innov.transpose() * S.ldlt().solve(innov))(0);
}

inline double normal_quantile(double p) {
  // Acklam inverse-normal approximation.
  static constexpr double a1 = -3.969683028665376e+01;
  static constexpr double a2 = 2.209460984245205e+02;
  static constexpr double a3 = -2.759285104469687e+02;
  static constexpr double a4 = 1.383577518672690e+02;
  static constexpr double a5 = -3.066479806614716e+01;
  static constexpr double a6 = 2.506628277459239e+00;

  static constexpr double b1 = -5.447609879822406e+01;
  static constexpr double b2 = 1.615858368580409e+02;
  static constexpr double b3 = -1.556989798598866e+02;
  static constexpr double b4 = 6.680131188771972e+01;
  static constexpr double b5 = -1.328068155288572e+01;

  static constexpr double c1 = -7.784894002430293e-03;
  static constexpr double c2 = -3.223964580411365e-01;
  static constexpr double c3 = -2.400758277161838e+00;
  static constexpr double c4 = -2.549732539343734e+00;
  static constexpr double c5 = 4.374664141464968e+00;
  static constexpr double c6 = 2.938163982698783e+00;

  static constexpr double d1 = 7.784695709041462e-03;
  static constexpr double d2 = 3.224671290700398e-01;
  static constexpr double d3 = 2.445134137142996e+00;
  static constexpr double d4 = 3.754408661907416e+00;

  if (p <= 0.0) {
    return -std::numeric_limits<double>::infinity();
  }
  if (p >= 1.0) {
    return std::numeric_limits<double>::infinity();
  }

  const double plow = 0.02425;
  const double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
         (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

inline double chi_square_quantile_approx(double p, int dof) {
  if (dof <= 0) {
    return 0.0;
  }
  const double k = static_cast<double>(dof);
  const double z = normal_quantile(p);
  const double a = 2.0 / (9.0 * k);
  const double w = 1.0 - a + z * std::sqrt(a);
  return k * w * w * w;
}

struct ChiSquareGate {
  double lower = 0.0;
  double upper = 0.0;
  bool in_gate = true;
};

inline ChiSquareGate chi_square_gate_for_average(double value,
                                                 int samples,
                                                 int dof_per_sample,
                                                 double alpha = 0.05) {
  const int dof_total = samples * dof_per_sample;
  if (samples <= 0 || dof_per_sample <= 0) {
    return {0.0, 0.0, false};
  }
  const double lo = chi_square_quantile_approx(alpha * 0.5, dof_total) / static_cast<double>(samples);
  const double hi = chi_square_quantile_approx(1.0 - alpha * 0.5, dof_total) / static_cast<double>(samples);
  return {lo, hi, value >= lo && value <= hi};
}

struct ConsistencyAccumulator {
  int nis_count = 0;
  int nees_count = 0;
  double nis_sum = 0.0;
  double nees_sum = 0.0;

  void add_nis(double v) {
    ++nis_count;
    nis_sum += v;
  }

  void add_nees(double v) {
    ++nees_count;
    nees_sum += v;
  }

  double nis_mean() const {
    return nis_count > 0 ? nis_sum / static_cast<double>(nis_count) : 0.0;
  }

  double nees_mean() const {
    return nees_count > 0 ? nees_sum / static_cast<double>(nees_count) : 0.0;
  }

  ChiSquareGate nis_gate(int meas_dof, double alpha = 0.05) const {
    return chi_square_gate_for_average(nis_mean(), nis_count, meas_dof, alpha);
  }

  ChiSquareGate nees_gate(int state_dof, double alpha = 0.05) const {
    return chi_square_gate_for_average(nees_mean(), nees_count, state_dof, alpha);
  }
};

}  // namespace bierman::util
