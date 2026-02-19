#include <cmath>
#include <fstream>
#include <map>
#include <string>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/linalg/triangular.hpp"
#include "bierman/linalg/ud.hpp"
#include "bierman/models/ballistic_room.hpp"
#include "bierman/models/range_box_3d.hpp"
#include "bierman/util/rng.hpp"

namespace {

std::map<std::string, double> load_golden(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open golden metrics file: " + path);
  }
  std::map<std::string, double> out;
  std::string key;
  double val;
  while (in >> key >> val) {
    out[key] = val;
  }
  return out;
}

double rms3(const Eigen::Vector3d& acc, int n) {
  return std::sqrt(acc.squaredNorm() / static_cast<double>(n));
}

}  // namespace

int main() {
  auto golden = load_golden("tests/golden/metrics_baseline.txt");

  // Static case (deterministic with seed)
  {
    const unsigned int seed = 7;
    const int steps = 80;
    const double sigma_range = 0.25;

    Eigen::MatrixXd trackers(8, 3);
    trackers << 0, 0, 0,
        0, 0, 10,
        0, 10, 0,
        0, 10, 10,
        10, 0, 0,
        10, 0, 10,
        10, 10, 0,
        10, 10, 10;

    bierman::models::RangeOnlyBox3D model(trackers, sigma_range);
    bierman::util::Rng rng(seed);

    Eigen::Vector3d truth(4.3, 6.1, 7.7);
    Eigen::Vector3d x0(2.0, 2.0, 2.0);
    Eigen::Matrix3d P0 = 25.0 * Eigen::Matrix3d::Identity();

    bierman::filters::KalmanState kf{x0, P0};
    auto udf = bierman::linalg::ud_factorize(P0);
    bierman::filters::UDState ud{x0, udf.U, udf.d};

    Eigen::Matrix3d F0 = P0.ldlt().solve(Eigen::Matrix3d::Identity());
    Eigen::Matrix3d R0 = F0.llt().matrixU();
    bierman::filters::SRIFState srif{R0, R0 * x0};

    Eigen::Vector3d err_kf = Eigen::Vector3d::Zero();
    Eigen::Vector3d err_ud = Eigen::Vector3d::Zero();
    Eigen::Vector3d err_srif = Eigen::Vector3d::Zero();

    const Eigen::MatrixXd sqrtW = (1.0 / sigma_range) * Eigen::MatrixXd::Identity(trackers.rows(), trackers.rows());

    for (int k = 0; k < steps; ++k) {
      Eigen::VectorXd y = model.predict(truth);
      for (Eigen::Index i = 0; i < y.size(); ++i) {
        y(i) += rng.normal(0.0, sigma_range);
      }

      for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
        Eigen::VectorXd yhat_kf = model.predict(kf.x);
        Eigen::MatrixXd Hkf = model.jacobian(kf.x);
        bierman::filters::KalmanJoseph::update_scalar(kf, Hkf.row(i), y(i) - yhat_kf(i), 1.0 / sigma_range);

        Eigen::VectorXd yhat_u = model.predict(ud.x);
        Eigen::MatrixXd Hu = model.jacobian(ud.x);
        bierman::filters::UDFilter::update_scalar(ud, Hu.row(i), y(i) - yhat_u(i), sigma_range * sigma_range);
      }

      Eigen::VectorXd xs = bierman::linalg::solve_upper(srif.R, srif.z);
      Eigen::VectorXd yhs = model.predict(xs);
      Eigen::MatrixXd Hs = model.jacobian(xs);
      Eigen::VectorXd zlin = y - yhs + Hs * xs;
      srif = bierman::filters::SRIF::update_householder(srif, Hs, zlin, sqrtW).state;
      xs = bierman::linalg::solve_upper(srif.R, srif.z);

      err_kf += (kf.x - truth).cwiseAbs2();
      err_ud += (ud.x - truth).cwiseAbs2();
      err_srif += (xs - truth).cwiseAbs2();
    }

    const double kf_rms = rms3(err_kf, steps);
    const double ud_rms = rms3(err_ud, steps);
    const double srif_rms = rms3(err_srif, steps);

    if (std::abs(kf_rms - golden["static_kf_rms"]) > 1e-8 ||
        std::abs(ud_rms - golden["static_ud_rms"]) > 1e-8 ||
        std::abs(srif_rms - golden["static_srif_rms"]) > 1e-8) {
      spdlog::error("Static golden drift: kf={} ud={} srif={}", kf_rms, ud_rms, srif_rms);
      return 1;
    }
  }

  // Dynamic case
  {
    const unsigned int seed = 11;
    const double dt = 0.1;
    const int steps = 120;
    const double sigma_range = 0.2;

    Eigen::MatrixXd trackers(8, 3);
    trackers << 0, 0, 0,
        0, 0, 10,
        0, 10, 0,
        0, 10, 10,
        10, 0, 0,
        10, 0, 10,
        10, 10, 0,
        10, 10, 10;

    bierman::models::BallisticRoom dyn_truth(0.06, Eigen::Vector3d(0.0, 0.0, -9.81));
    bierman::models::BallisticRoom dyn_filter(0.0, Eigen::Vector3d(0.0, 0.0, -9.81));
    bierman::util::Rng rng(seed);

    Eigen::VectorXd truth(6);
    truth << 3.0, 4.0, 8.0, 1.2, -0.3, 0.5;

    Eigen::VectorXd x0 = truth;
    x0.head<3>() += Eigen::Vector3d(0.5, -0.8, 0.3);
    x0.tail<3>() += Eigen::Vector3d(0.2, -0.2, 0.1);

    Eigen::MatrixXd P0 = 5.0 * Eigen::MatrixXd::Identity(6, 6);
    Eigen::MatrixXd Q = 1e-3 * Eigen::MatrixXd::Identity(3, 3);

    bierman::filters::KalmanState kf{x0, P0};
    auto udf = bierman::linalg::ud_factorize(P0);
    bierman::filters::UDState ud{x0, udf.U, udf.d};

    Eigen::Vector3d err_kf = Eigen::Vector3d::Zero();
    Eigen::Vector3d err_ud = Eigen::Vector3d::Zero();

    for (int k = 0; k < steps; ++k) {
      truth = dyn_truth.propagate_truth(dt, truth);

      Eigen::VectorXd y(trackers.rows());
      for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
        y(i) = (truth.head<3>() - trackers.row(i).transpose()).norm() + rng.normal(0.0, sigma_range);
      }

      kf.x = dyn_filter.propagate_filter(dt, kf.x);
      Eigen::MatrixXd Phi = dyn_filter.state_transition(dt);
      Eigen::MatrixXd G = dyn_filter.process_noise_jacobian(dt);
      kf.P = Phi * kf.P * Phi.transpose() + G * Q * G.transpose();

      ud.x = dyn_filter.propagate_filter(dt, ud.x);
      bierman::filters::UDFilter::predict_diagonal_q(ud, Phi, Q.diagonal(), G);

      for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
        Eigen::Vector3d dk = kf.x.head<3>() - trackers.row(i).transpose();
        Eigen::Vector3d du = ud.x.head<3>() - trackers.row(i).transpose();
        const double rk = dk.norm();
        const double ru = du.norm();
        Eigen::RowVectorXd Hk = Eigen::RowVectorXd::Zero(6);
        Eigen::RowVectorXd Hu = Eigen::RowVectorXd::Zero(6);
        if (rk > 0.0) {
          Hk.head<3>() = (dk / rk).transpose();
        }
        if (ru > 0.0) {
          Hu.head<3>() = (du / ru).transpose();
        }
        bierman::filters::KalmanJoseph::update_scalar(kf, Hk, y(i) - rk, 1.0 / sigma_range);
        bierman::filters::UDFilter::update_scalar(ud, Hu, y(i) - ru, sigma_range * sigma_range);
      }

      err_kf += (kf.x.head<3>() - truth.head<3>()).cwiseAbs2();
      err_ud += (ud.x.head<3>() - truth.head<3>()).cwiseAbs2();
    }

    const double kf_rms = rms3(err_kf, steps);
    const double ud_rms = rms3(err_ud, steps);

    if (std::abs(kf_rms - golden["dynamic_kf_rms"]) > 1e-8 ||
        std::abs(ud_rms - golden["dynamic_ud_rms"]) > 1e-8) {
      spdlog::error("Dynamic golden drift: kf={} ud={}", kf_rms, ud_rms);
      return 1;
    }
  }

  return 0;
}
