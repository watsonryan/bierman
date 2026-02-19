#include <filesystem>
#include <fstream>
#include <string>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/schmidt.hpp"
#include "bierman/filters/srukf.hpp"
#include "bierman/filters/ukf.hpp"
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

}  // namespace

int main(int argc, char** argv) {
  const unsigned int seed = get_arg_u(argc, argv, "--seed", 19);
  const double dt = get_arg(argc, argv, "--dt", 0.1);
  const double tf = get_arg(argc, argv, "--tf", 15.0);
  const double sigma_range = get_arg(argc, argv, "--sigma_range", 0.3);
  const int steps = static_cast<int>(tf / dt);
  const std::string outdir = "output_ukf_srukf_bias";

  std::filesystem::create_directories(outdir);
  bierman::util::Rng rng(seed);

  Eigen::MatrixXd trackers(4, 3);
  trackers << 0, 0, 0,
      10, 0, 0,
      0, 10, 0,
      0, 0, 10;

  const int nx = 6;
  const int nb = trackers.rows();
  const int naug = nx + nb;

  Eigen::VectorXd truth(nx);
  truth << 2, 2, 7, 1.1, -0.4, 0.2;
  Eigen::VectorXd bias_true(nb);
  bias_true << 0.25, -0.10, 0.15, -0.20;

  auto propagate = [dt](const Eigen::VectorXd& x) {
    Eigen::VectorXd xn = x;
    xn(0) += dt * x(3);
    xn(1) += dt * x(4);
    xn(2) += dt * x(5);
    return xn;
  };

  auto measure_aug = [&trackers, nx](const Eigen::VectorXd& x) {
    Eigen::VectorXd y(trackers.rows());
    for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
      const double r = (x.head<3>() - trackers.row(i).transpose()).norm();
      y(i) = r + x(nx + i);
    }
    return y;
  };

  bierman::filters::UKFState ukf_state;
  ukf_state.x = Eigen::VectorXd::Zero(naug);
  ukf_state.x.head(nx) = truth + Eigen::VectorXd::Constant(nx, 0.3);
  ukf_state.P = 2.0 * Eigen::MatrixXd::Identity(naug, naug);

  bierman::filters::SRUKFState srukf_state;
  srukf_state.x = ukf_state.x;
  srukf_state.L = ukf_state.P.llt().matrixL();

  bierman::filters::SchmidtState sch;
  sch.x = ukf_state.x.head(nx);
  sch.Px = ukf_state.P.topLeftCorner(nx, nx);
  sch.Py = ukf_state.P.bottomRightCorner(nb, nb);
  sch.Pxy = ukf_state.P.topRightCorner(nx, nb);

  Eigen::MatrixXd Qaug = 1e-3 * Eigen::MatrixXd::Identity(naug, naug);
  Qaug.bottomRightCorner(nb, nb).setZero();
  Eigen::MatrixXd R = sigma_range * sigma_range * Eigen::MatrixXd::Identity(nb, nb);

  std::ofstream ftruth(outdir + "/truth.csv");
  std::ofstream fukf(outdir + "/estimates_ukf.csv");
  std::ofstream fsrukf(outdir + "/estimates_srukf.csv");
  std::ofstream fsch(outdir + "/estimates_schmidt.csv");

  bierman::util::write_csv_header(ftruth, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fukf, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fsrukf, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fsch, {"t", "x", "y", "z"});

  bierman::filters::UTParams utp{1e-1, 0.0, 2.0};

  for (int k = 0; k < steps; ++k) {
    const double t = k * dt;
    truth(0) += dt * truth(3);
    truth(1) += dt * truth(4);
    truth(2) += dt * truth(5);

    Eigen::VectorXd y(nb);
    for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
      y(i) = (truth.head<3>() - trackers.row(i).transpose()).norm() + bias_true(i) + rng.normal(0.0, sigma_range);
    }

    auto ukf_pred = bierman::filters::predict(ukf_state, Qaug, utp, propagate);
    ukf_state = bierman::filters::update(ukf_pred, y, R, measure_aug);

    Eigen::MatrixXd Lq = Qaug.llt().matrixL();
    Eigen::MatrixXd Lr = R.llt().matrixL();
    auto sr_pred = bierman::filters::SRUKF::predict(srukf_state, Lq, utp, propagate);
    srukf_state = bierman::filters::SRUKF::update(sr_pred, y, Lr, measure_aug, nb);

    Eigen::MatrixXd Phi = Eigen::MatrixXd::Identity(nx, nx);
    Phi.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
    sch.x = Phi * sch.x;
    sch.Px = Phi * sch.Px * Phi.transpose() + 1e-3 * Eigen::MatrixXd::Identity(nx, nx);
    sch.Pxy = Phi * sch.Pxy;

    Eigen::VectorXd yhat(nb);
    Eigen::MatrixXd Ax = Eigen::MatrixXd::Zero(nb, nx);
    Eigen::MatrixXd Ay = Eigen::MatrixXd::Identity(nb, nb);
    for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
      Eigen::Vector3d d = sch.x.head<3>() - trackers.row(i).transpose();
      double rr = d.norm();
      yhat(i) = rr;
      if (rr > 0.0) {
        Ax.block<1, 3>(i, 0) = (d / rr).transpose();
      }
    }
    bierman::filters::SchmidtKalman::update(sch, Ax, Ay, y - yhat, R);

    bierman::util::write_csv_row(ftruth, {t, truth(0), truth(1), truth(2)});
    bierman::util::write_csv_row(fukf, {t, ukf_state.x(0), ukf_state.x(1), ukf_state.x(2)});
    bierman::util::write_csv_row(fsrukf, {t, srukf_state.x(0), srukf_state.x(1), srukf_state.x(2)});
    bierman::util::write_csv_row(fsch, {t, sch.x(0), sch.x(1), sch.x(2)});
  }

  spdlog::info("Wrote UKF/SRUKF/Schmidt bias comparison CSV files to {}", outdir);
  return 0;
}
