#include <filesystem>
#include <fstream>
#include <string>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/linalg/ud.hpp"
#include "bierman/linalg/triangular.hpp"
#include "bierman/models/ballistic_room.hpp"
#include "bierman/util/csv.hpp"
#include "bierman/util/rng.hpp"

namespace {

double get_arg(int argc, char** argv, const std::string& key, double def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (key == argv[i]) {
      return std::stod(argv[i + 1]);
    }
  }
  return def;
}

unsigned int get_arg_u(int argc, char** argv, const std::string& key, unsigned int def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (key == argv[i]) {
      return static_cast<unsigned int>(std::stoul(argv[i + 1]));
    }
  }
  return def;
}

Eigen::VectorXd range_predict(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::MatrixXd>& trackers) {
  Eigen::VectorXd y(trackers.rows());
  for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
    y(i) = (x.head<3>() - trackers.row(i).transpose()).norm();
  }
  return y;
}

Eigen::MatrixXd range_jac(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::MatrixXd>& trackers) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(trackers.rows(), 6);
  for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
    Eigen::Vector3d d = x.head<3>() - trackers.row(i).transpose();
    const double r = d.norm();
    if (r > 0.0) {
      H.block<1, 3>(i, 0) = (d / r).transpose();
    }
  }
  return H;
}

}  // namespace

int main(int argc, char** argv) {
  const unsigned int seed = get_arg_u(argc, argv, "--seed", 11);
  const double dt = get_arg(argc, argv, "--dt", 0.1);
  const double tf = get_arg(argc, argv, "--tf", 20.0);
  const double sigma_range = get_arg(argc, argv, "--sigma_range", 0.2);
  const int steps = static_cast<int>(tf / dt);
  const std::string outdir = "output_dynamic";

  std::filesystem::create_directories(outdir);

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

  Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(6, 6));
  Eigen::MatrixXd R0 = F0.llt().matrixU();
  bierman::filters::SRIFState srif{R0, R0 * x0};

  std::ofstream ftruth(outdir + "/truth.csv");
  std::ofstream fkf(outdir + "/estimates_kalman.csv");
  std::ofstream fud(outdir + "/estimates_ud.csv");
  std::ofstream fsrif(outdir + "/estimates_srif.csv");

  bierman::util::write_csv_header(ftruth, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fkf, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fud, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fsrif, {"t", "x", "y", "z"});

  const Eigen::MatrixXd sqrtW = (1.0 / sigma_range) * Eigen::MatrixXd::Identity(trackers.rows(), trackers.rows());

  for (int k = 0; k < steps; ++k) {
    const double t = k * dt;
    truth = dyn_truth.propagate_truth(dt, truth);

    Eigen::VectorXd y = range_predict(truth, trackers);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      y(i) += rng.normal(0.0, sigma_range);
    }

    kf.x = dyn_filter.propagate_filter(dt, kf.x);
    Eigen::MatrixXd Phi = dyn_filter.state_transition(dt);
    Eigen::MatrixXd G = dyn_filter.process_noise_jacobian(dt);
    kf.P = Phi * kf.P * Phi.transpose() + G * Q * G.transpose();

    ud.x = dyn_filter.propagate_filter(dt, ud.x);
    ud.U.diagonal().setOnes();
    ud.U = ud.U.triangularView<Eigen::Upper>();
    ud.d = ud.d.cwiseMax(1e-12);
    bierman::filters::UDFilter::predict_diagonal_q(ud, Phi, Q.diagonal(), G);

    Eigen::MatrixXd PhiInv = Phi.ldlt().solve(Eigen::MatrixXd::Identity(Phi.rows(), Phi.cols()));
    Eigen::MatrixXd Lq = Q.llt().matrixL();
    Eigen::MatrixXd Rw = Lq.ldlt().solve(Eigen::MatrixXd::Identity(Lq.rows(), Lq.cols()));
    srif = bierman::filters::SRIF::predict(srif, PhiInv, Rw, G);

    for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
      Eigen::VectorXd yhat = range_predict(kf.x, trackers);
      Eigen::MatrixXd H = range_jac(kf.x, trackers);
      bierman::filters::KalmanJoseph::update_scalar(kf, H.row(i), y(i) - yhat(i), 1.0 / sigma_range);

      Eigen::VectorXd yhu = range_predict(ud.x, trackers);
      Eigen::MatrixXd Hu = range_jac(ud.x, trackers);
      bierman::filters::UDFilter::update_scalar(ud, Hu.row(i), y(i) - yhu(i), sigma_range * sigma_range);
    }

    Eigen::VectorXd xs = bierman::linalg::solve_upper(srif.R, srif.z);
    Eigen::VectorXd yhs = range_predict(xs, trackers);
    Eigen::MatrixXd Hs = range_jac(xs, trackers);
    Eigen::VectorXd zlin = y - yhs + Hs * xs;
    auto su = bierman::filters::SRIF::update_householder(srif, Hs, zlin, sqrtW);
    srif = su.state;
    xs = bierman::linalg::solve_upper(srif.R, srif.z);

    bierman::util::write_csv_row(ftruth, {t, truth(0), truth(1), truth(2)});
    bierman::util::write_csv_row(fkf, {t, kf.x(0), kf.x(1), kf.x(2)});
    bierman::util::write_csv_row(fud, {t, ud.x(0), ud.x(1), ud.x(2)});
    bierman::util::write_csv_row(fsrif, {t, xs(0), xs(1), xs(2)});
  }

  spdlog::info("Wrote dynamic comparison CSV files to {}", outdir);
  return 0;
}
