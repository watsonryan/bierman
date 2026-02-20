#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <CLI/CLI.hpp>
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/linalg/ud.hpp"
#include "bierman/linalg/triangular.hpp"
#include "bierman/models/ballistic_room.hpp"
#include "bierman/util/csv.hpp"
#include "bierman/util/rng.hpp"
#include "bierman/util/stats.hpp"

namespace {

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
  unsigned int seed = 11;
  double dt = 0.1;
  double tf = 20.0;
  double sigma_range = 0.2;
  std::string outdir = "output_dynamic";
  bool quiet = false;
  bool iterated = false;
  int iter_max = 4;

  CLI::App app{"Dynamic range-only filter comparison"};
  app.set_config("--config", "", "INI/TOML config file path");
  app.add_option("--seed", seed, "RNG seed");
  app.add_option("--dt", dt, "Step size");
  app.add_option("--tf", tf, "Final time");
  app.add_option("--sigma_range", sigma_range, "Range measurement sigma");
  app.add_option("--outdir", outdir, "Output directory");
  app.add_flag("--quiet", quiet, "Suppress info logging");
  app.add_flag("--iterated", iterated, "Use iterated nonlinear measurement updates");
  app.add_option("--iter-max", iter_max, "Maximum iterated-update passes");
  CLI11_PARSE(app, argc, argv);

  spdlog::set_level(quiet ? spdlog::level::warn : spdlog::level::info);

  const int steps = static_cast<int>(tf / dt);
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

  const Eigen::MatrixXd H0 = range_jac(x0, trackers).leftCols(3);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H0, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd svals = svd.singularValues();
  const double cond0 = svals(svals.size() - 1) > 0.0 ? svals(0) / svals(svals.size() - 1) : std::numeric_limits<double>::infinity();
  if (!std::isfinite(cond0) || cond0 > 1e8) {
    spdlog::warn("Weak initial geometry conditioning in dynamic case: cond={}", cond0);
  }

  Eigen::MatrixXd P0 = 5.0 * Eigen::MatrixXd::Identity(6, 6);
  Eigen::MatrixXd Q = 1e-3 * Eigen::MatrixXd::Identity(3, 3);

  bierman::filters::KalmanState kf{x0, P0};
  auto udf = bierman::linalg::ud_factorize(P0);
  bierman::filters::UDState ud{x0, udf.U, udf.d};

  Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(6, 6));
  Eigen::MatrixXd R0 = F0.llt().matrixU();
  bierman::filters::SRIFState srif{R0, R0 * x0};

  bierman::util::ConsistencyAccumulator kf_stats;
  int srif_iter_fallbacks = 0;

  std::ofstream ftruth(outdir + "/truth.csv");
  std::ofstream fkf(outdir + "/estimates_kalman.csv");
  std::ofstream fud(outdir + "/estimates_ud.csv");
  std::ofstream fsrif(outdir + "/estimates_srif.csv");
  std::ofstream fsum(outdir + "/summary_metrics.csv");

  bierman::util::write_csv_header(ftruth, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fkf, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fud, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fsrif, {"t", "x", "y", "z"});
  bierman::util::write_csv_header(fsum, {"metric", "value"});

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
      const double innov = y(i) - yhat(i);
      const double S = (H.row(i) * kf.P * H.row(i).transpose())(0) + sigma_range * sigma_range;
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
      bierman::filters::KalmanJoseph::update_scalar(kf_single, H.row(i), innov, 1.0 / sigma_range);

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

      Eigen::VectorXd yhu = range_predict(ud.x, trackers);
      Eigen::MatrixXd Hu = range_jac(ud.x, trackers);
      bierman::filters::UDFilter::update_scalar(ud, Hu.row(i), y(i) - yhu(i), sigma_range * sigma_range);
    }

    {
      const bierman::filters::SRIFState srif_prior = srif;
      const Eigen::VectorXd x_prior = bierman::linalg::solve_upper(srif_prior.R, srif_prior.z);
      const Eigen::VectorXd yhat_prior = range_predict(x_prior, trackers);
      const Eigen::MatrixXd H_prior = range_jac(x_prior, trackers);
      const Eigen::VectorXd zlin = y - yhat_prior + H_prior * x_prior;
      const auto single = bierman::filters::SRIF::update_householder(srif_prior, H_prior, zlin, sqrtW);

      if (iterated) {
        ++srif_iter_fallbacks;
      }
      srif = single.state;
    }

    kf_stats.add_nees(bierman::util::nees(kf.x.head<3>() - truth.head<3>(), kf.P.topLeftCorner(3, 3)));

    Eigen::VectorXd xs = bierman::linalg::solve_upper(srif.R, srif.z);

    bierman::util::write_csv_row(ftruth, {t, truth(0), truth(1), truth(2)});
    bierman::util::write_csv_row(fkf, {t, kf.x(0), kf.x(1), kf.x(2)});
    bierman::util::write_csv_row(fud, {t, ud.x(0), ud.x(1), ud.x(2)});
    bierman::util::write_csv_row(fsrif, {t, xs(0), xs(1), xs(2)});
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

  spdlog::info("Wrote dynamic comparison CSV files to {}", outdir);
  spdlog::info("KF NIS mean={} gate=[{}, {}] pass={}", kf_stats.nis_mean(), nis_gate.lower, nis_gate.upper, nis_gate.in_gate);
  spdlog::info("KF NEES mean={} gate=[{}, {}] pass={}", kf_stats.nees_mean(), nees_gate.lower, nees_gate.upper, nees_gate.in_gate);
  if (iterated) {
    spdlog::info("SRIF iterated disabled steps: {}", srif_iter_fallbacks);
  }
  return 0;
}
