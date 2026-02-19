#include <filesystem>
#include <fstream>
#include <string>

#include <CLI/CLI.hpp>
#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/potter_sr.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/linalg/triangular.hpp"
#include "bierman/linalg/ud.hpp"
#include "bierman/models/range_box_3d.hpp"
#include "bierman/util/csv.hpp"
#include "bierman/util/rng.hpp"

int main(int argc, char** argv) {
  unsigned int seed = 7;
  int steps = 100;
  double sigma_range = 0.25;
  std::string outdir = "output_static";
  bool quiet = false;

  CLI::App app{"Static 3D range-only filter comparison"};
  std::string config_file;
  app.set_config("--config", "", "INI/TOML config file path");
  app.add_option("--seed", seed, "RNG seed");
  app.add_option("--steps", steps, "Number of sequential updates");
  app.add_option("--sigma_range", sigma_range, "Range measurement sigma");
  app.add_option("--outdir", outdir, "Output directory");
  app.add_flag("--quiet", quiet, "Suppress info logging");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(quiet ? spdlog::level::warn : spdlog::level::info);

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

  bierman::models::RangeOnlyBox3D model(trackers, sigma_range);
  bierman::util::Rng rng(seed);

  Eigen::Vector3d truth(4.3, 6.1, 7.7);
  Eigen::Vector3d x0(2.0, 2.0, 2.0);
  Eigen::Matrix3d P0 = 25.0 * Eigen::Matrix3d::Identity();

  bierman::filters::KalmanState kf{x0, P0};
  bierman::filters::PotterState pot{x0, P0.llt().matrixL()};

  const auto ud_fac = bierman::linalg::ud_factorize(P0);
  bierman::filters::UDState ud{x0, ud_fac.U, ud_fac.d};

  Eigen::Matrix3d F0 = P0.ldlt().solve(Eigen::Matrix3d::Identity());
  Eigen::Matrix3d R0 = F0.llt().matrixU();
  bierman::filters::SRIFState srif_hh{R0, R0 * x0};
  bierman::filters::SRIFState srif_qr{R0, R0 * x0};

  std::ofstream ftruth(outdir + "/truth.csv");
  std::ofstream fkf(outdir + "/estimates_kalman.csv");
  std::ofstream fpot(outdir + "/estimates_potter.csv");
  std::ofstream fud(outdir + "/estimates_ud.csv");
  std::ofstream fhh(outdir + "/estimates_srif_hh.csv");
  std::ofstream fqr(outdir + "/estimates_srif_qr.csv");

  bierman::util::write_csv_header(ftruth, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fkf, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fpot, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fud, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fhh, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fqr, {"k", "x", "y", "z"});

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

      Eigen::VectorXd yhat_p = model.predict(pot.x);
      Eigen::MatrixXd Hp = model.jacobian(pot.x);
      bierman::filters::PotterSR::update_scalar(pot, Hp.row(i), y(i) - yhat_p(i), 1.0 / sigma_range);

      Eigen::VectorXd yhat_u = model.predict(ud.x);
      Eigen::MatrixXd Hu = model.jacobian(ud.x);
      bierman::filters::UDFilter::update_scalar(ud, Hu.row(i), y(i) - yhat_u(i), sigma_range * sigma_range);
    }

    Eigen::VectorXd xhhv = bierman::linalg::solve_upper(srif_hh.R, srif_hh.z);
    Eigen::VectorXd yhat_hh = model.predict(xhhv);
    Eigen::MatrixXd Hh = model.jacobian(xhhv);
    Eigen::VectorXd zlin_hh = y - yhat_hh + Hh * xhhv;
    auto uh = bierman::filters::SRIF::update_householder(srif_hh, Hh, zlin_hh, sqrtW);
    srif_hh = uh.state;

    Eigen::VectorXd xqrv = bierman::linalg::solve_upper(srif_qr.R, srif_qr.z);
    Eigen::VectorXd yhat_qr = model.predict(xqrv);
    Eigen::MatrixXd Hq = model.jacobian(xqrv);
    Eigen::VectorXd zlin_qr = y - yhat_qr + Hq * xqrv;
    auto uq = bierman::filters::SRIF::update_qr_mgs(srif_qr, Hq, zlin_qr, sqrtW);
    srif_qr = uq.state;

    const Eigen::Vector3d xhh = bierman::linalg::solve_upper(srif_hh.R, srif_hh.z);
    const Eigen::Vector3d xqr = bierman::linalg::solve_upper(srif_qr.R, srif_qr.z);

    bierman::util::write_csv_row(ftruth, {static_cast<double>(k), truth(0), truth(1), truth(2)});
    bierman::util::write_csv_row(fkf, {static_cast<double>(k), kf.x(0), kf.x(1), kf.x(2)});
    bierman::util::write_csv_row(fpot, {static_cast<double>(k), pot.x(0), pot.x(1), pot.x(2)});
    bierman::util::write_csv_row(fud, {static_cast<double>(k), ud.x(0), ud.x(1), ud.x(2)});
    bierman::util::write_csv_row(fhh, {static_cast<double>(k), xhh(0), xhh(1), xhh(2)});
    bierman::util::write_csv_row(fqr, {static_cast<double>(k), xqr(0), xqr(1), xqr(2)});
  }

  spdlog::info("Wrote static comparison CSV files to {}", outdir);
  return 0;
}
