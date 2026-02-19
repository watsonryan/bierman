#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

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
#include "bierman/util/stats.hpp"

int main(int argc, char** argv) {
  unsigned int seed = 7;
  int steps = 100;
  double sigma_range = 0.25;
  std::string outdir = "output_static";
  bool quiet = false;
  bool iterated = false;
  int iter_max = 4;

  CLI::App app{"Static 3D range-only filter comparison"};
  app.set_config("--config", "", "INI/TOML config file path");
  app.add_option("--seed", seed, "RNG seed");
  app.add_option("--steps", steps, "Number of sequential updates");
  app.add_option("--sigma_range", sigma_range, "Range measurement sigma");
  app.add_option("--outdir", outdir, "Output directory");
  app.add_flag("--quiet", quiet, "Suppress info logging");
  app.add_flag("--iterated", iterated, "Use iterated nonlinear measurement updates");
  app.add_option("--iter-max", iter_max, "Maximum iterated-update passes");
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

  const auto obs = model.observability(x0);
  if (!obs.well_conditioned()) {
    spdlog::warn("Weak observability at initial point: rank={}, cond={}, sigma_min={}",
                 obs.rank,
                 obs.condition,
                 obs.sigma_min);
  }

  bierman::filters::KalmanState kf{x0, P0};
  bierman::filters::PotterState pot{x0, P0.llt().matrixL()};

  const auto ud_fac = bierman::linalg::ud_factorize(P0);
  bierman::filters::UDState ud{x0, ud_fac.U, ud_fac.d};

  Eigen::Matrix3d F0 = P0.ldlt().solve(Eigen::Matrix3d::Identity());
  Eigen::Matrix3d R0 = F0.llt().matrixU();
  bierman::filters::SRIFState srif_hh{R0, R0 * x0};
  bierman::filters::SRIFState srif_qr{R0, R0 * x0};

  bierman::util::ConsistencyAccumulator kf_stats;
  int srif_iter_fallbacks = 0;

  std::ofstream ftruth(outdir + "/truth.csv");
  std::ofstream fkf(outdir + "/estimates_kalman.csv");
  std::ofstream fpot(outdir + "/estimates_potter.csv");
  std::ofstream fud(outdir + "/estimates_ud.csv");
  std::ofstream fhh(outdir + "/estimates_srif_hh.csv");
  std::ofstream fqr(outdir + "/estimates_srif_qr.csv");
  std::ofstream fsum(outdir + "/summary_metrics.csv");

  bierman::util::write_csv_header(ftruth, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fkf, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fpot, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fud, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fhh, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fqr, {"k", "x", "y", "z"});
  bierman::util::write_csv_header(fsum, {"metric", "value"});

  const Eigen::MatrixXd sqrtW = (1.0 / sigma_range) * Eigen::MatrixXd::Identity(trackers.rows(), trackers.rows());

  for (int k = 0; k < steps; ++k) {
    Eigen::VectorXd y = model.predict(truth);
    for (Eigen::Index i = 0; i < y.size(); ++i) {
      y(i) += rng.normal(0.0, sigma_range);
    }

    for (Eigen::Index i = 0; i < trackers.rows(); ++i) {
      Eigen::VectorXd yhat_kf = model.predict(kf.x);
      Eigen::MatrixXd Hkf = model.jacobian(kf.x);
      const double innov = y(i) - yhat_kf(i);
      const double S = (Hkf.row(i) * kf.P * Hkf.row(i).transpose())(0) + sigma_range * sigma_range;
      kf_stats.add_nis((innov * innov) / S);

      const Eigen::Vector3d tracker = trackers.row(i).transpose();
      const auto h_tracker = [tracker](const Eigen::VectorXd& x) { return (x.head<3>() - tracker).norm(); };
      const auto H_tracker = [tracker](const Eigen::VectorXd& x) {
        Eigen::RowVectorXd Hloc = Eigen::RowVectorXd::Zero(x.size());
        Eigen::Vector3d d = x.head<3>() - tracker;
        const double r = d.norm();
        if (r > 0.0) {
          Hloc.head<3>() = (d / r).transpose();
        }
        return Hloc;
      };

      bierman::filters::KalmanState kf_single = kf;
      bierman::filters::KalmanJoseph::update_scalar(kf_single, Hkf.row(i), innov, 1.0 / sigma_range);

      if (iterated) {
        bierman::filters::KalmanState kf_iter = kf;
        bierman::filters::KalmanJoseph::update_scalar_iterated(
            kf_iter,
            y(i),
            1.0 / sigma_range,
            h_tracker,
            H_tracker,
            iter_max);

        const Eigen::LDLT<Eigen::MatrixXd> ldlt(kf.P);
        const auto objective = [&](const bierman::filters::KalmanState& s) {
          const Eigen::VectorXd dx = s.x - kf.x;
          const double prior_cost = dx.dot(ldlt.solve(dx));
          const double meas_cost = std::pow((y(i) - h_tracker(s.x)) / sigma_range, 2);
          return std::pair<double, double>{prior_cost + meas_cost, prior_cost};
        };

        const auto [single_cost, single_step2] = objective(kf_single);
        const auto [iter_cost, iter_step2] = objective(kf_iter);
        constexpr double kMaxPriorStep2 = 25.0;

        if (iter_cost <= 0.999 * single_cost && iter_step2 <= kMaxPriorStep2) {
          kf = std::move(kf_iter);
        } else {
          kf = std::move(kf_single);
        }
      } else {
        kf = std::move(kf_single);
      }

      Eigen::VectorXd yhat_p = model.predict(pot.x);
      Eigen::MatrixXd Hp = model.jacobian(pot.x);
      bierman::filters::PotterSR::update_scalar(pot, Hp.row(i), y(i) - yhat_p(i), 1.0 / sigma_range);

      Eigen::VectorXd yhat_u = model.predict(ud.x);
      Eigen::MatrixXd Hu = model.jacobian(ud.x);
      bierman::filters::UDFilter::update_scalar(ud, Hu.row(i), y(i) - yhat_u(i), sigma_range * sigma_range);
    }

    {
      const bierman::filters::SRIFState srif_prior = srif_hh;
      const Eigen::VectorXd x_prior = bierman::linalg::solve_upper(srif_prior.R, srif_prior.z);
      const Eigen::VectorXd yhat_prior = model.predict(x_prior);
      const Eigen::MatrixXd H_prior = model.jacobian(x_prior);
      const Eigen::VectorXd zlin_hh = y - yhat_prior + H_prior * x_prior;
      const auto single = bierman::filters::SRIF::update_householder(srif_prior, H_prior, zlin_hh, sqrtW);

      if (iterated) {
        ++srif_iter_fallbacks;
      }
      srif_hh = single.state;
    }

    Eigen::VectorXd xqrv = bierman::linalg::solve_upper(srif_qr.R, srif_qr.z);
    Eigen::VectorXd yhat_qr = model.predict(xqrv);
    Eigen::MatrixXd Hq = model.jacobian(xqrv);
    Eigen::VectorXd zlin_qr = y - yhat_qr + Hq * xqrv;
    srif_qr = bierman::filters::SRIF::update_qr_mgs(srif_qr, Hq, zlin_qr, sqrtW).state;

    const Eigen::Vector3d xhh = bierman::linalg::solve_upper(srif_hh.R, srif_hh.z);
    const Eigen::Vector3d xqr = bierman::linalg::solve_upper(srif_qr.R, srif_qr.z);

    kf_stats.add_nees(bierman::util::nees(kf.x - truth, kf.P));

    bierman::util::write_csv_row(ftruth, {static_cast<double>(k), truth(0), truth(1), truth(2)});
    bierman::util::write_csv_row(fkf, {static_cast<double>(k), kf.x(0), kf.x(1), kf.x(2)});
    bierman::util::write_csv_row(fpot, {static_cast<double>(k), pot.x(0), pot.x(1), pot.x(2)});
    bierman::util::write_csv_row(fud, {static_cast<double>(k), ud.x(0), ud.x(1), ud.x(2)});
    bierman::util::write_csv_row(fhh, {static_cast<double>(k), xhh(0), xhh(1), xhh(2)});
    bierman::util::write_csv_row(fqr, {static_cast<double>(k), xqr(0), xqr(1), xqr(2)});
  }

  const auto nis_gate = kf_stats.nis_gate(1);
  const auto nees_gate = kf_stats.nees_gate(3);

  bierman::util::write_csv_row(fsum, "kf_nis_mean", kf_stats.nis_mean());
  bierman::util::write_csv_row(fsum, "kf_nis_gate_lo", nis_gate.lower);
  bierman::util::write_csv_row(fsum, "kf_nis_gate_hi", nis_gate.upper);
  bierman::util::write_csv_row(fsum, "kf_nis_in_gate", nis_gate.in_gate ? 1.0 : 0.0);
  bierman::util::write_csv_row(fsum, "kf_nees_mean", kf_stats.nees_mean());
  bierman::util::write_csv_row(fsum, "kf_nees_gate_lo", nees_gate.lower);
  bierman::util::write_csv_row(fsum, "kf_nees_gate_hi", nees_gate.upper);
  bierman::util::write_csv_row(fsum, "kf_nees_in_gate", nees_gate.in_gate ? 1.0 : 0.0);

  spdlog::info("Wrote static comparison CSV files to {}", outdir);
  spdlog::info("KF NIS mean={} gate=[{}, {}] pass={}", kf_stats.nis_mean(), nis_gate.lower, nis_gate.upper, nis_gate.in_gate);
  spdlog::info("KF NEES mean={} gate=[{}, {}] pass={}", kf_stats.nees_mean(), nees_gate.lower, nees_gate.upper, nees_gate.in_gate);
  if (iterated) {
    spdlog::info("SRIF iterated disabled steps: {}", srif_iter_fallbacks);
  }
  return 0;
}
