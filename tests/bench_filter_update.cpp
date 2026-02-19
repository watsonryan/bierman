#include <atomic>
#include <chrono>
#include <cstdlib>
#include <new>

#include <spdlog/spdlog.h>

#include "bierman/filters/kalman_joseph.hpp"
#include "bierman/filters/srif.hpp"
#include "bierman/filters/srukf.hpp"
#include "bierman/filters/ud_filter.hpp"
#include "bierman/filters/ukf.hpp"
#include "bierman/linalg/ud.hpp"

namespace {

std::atomic<long long> g_allocs{0};

void* operator_new_impl(std::size_t sz) {
  if (void* p = std::malloc(sz)) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    return p;
  }
  throw std::bad_alloc();
}

}  // namespace

void* operator new(std::size_t sz) { return operator_new_impl(sz); }
void* operator new[](std::size_t sz) { return operator_new_impl(sz); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void* operator new(std::size_t sz, std::align_val_t) { return operator_new_impl(sz); }
void* operator new[](std::size_t sz, std::align_val_t) { return operator_new_impl(sz); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }

int main() {
  constexpr int n = 6;
  constexpr int m = 1;
  constexpr int iters = 20000;

  Eigen::VectorXd x0 = Eigen::VectorXd::Random(n);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
  Eigen::MatrixXd P0 = A * A.transpose() + 0.1 * Eigen::MatrixXd::Identity(n, n);
  Eigen::RowVectorXd H = Eigen::RowVectorXd::Random(n);
  const double sigma = 0.3;
  const double sw = 1.0 / sigma;

  auto measure_scalar = [&H](const Eigen::VectorXd& x) { return H.dot(x); };

  auto bench = [&](const char* name, auto&& fn) {
    g_allocs.store(0, std::memory_order_relaxed);
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double ups = static_cast<double>(iters) / sec;
    const long long allocs = g_allocs.load(std::memory_order_relaxed);
    spdlog::info("{}: updates_per_sec={:.2f}, allocs_total={}, allocs_per_update={:.4f}",
                 name,
                 ups,
                 allocs,
                 static_cast<double>(allocs) / static_cast<double>(iters));
  };

  bench("KalmanJoseph", [&] {
    bierman::filters::KalmanState kf{x0, P0};
    for (int i = 0; i < iters; ++i) {
      const double y = measure_scalar(kf.x) + 0.01 * std::sin(0.01 * static_cast<double>(i));
      bierman::filters::KalmanJoseph::update_scalar(kf, H, y - measure_scalar(kf.x), sw);
    }
  });

  bench("UDFilter", [&] {
    auto fac = bierman::linalg::ud_factorize(P0);
    bierman::filters::UDState ud{x0, fac.U, fac.d};
    for (int i = 0; i < iters; ++i) {
      const double y = measure_scalar(ud.x) + 0.01 * std::sin(0.01 * static_cast<double>(i));
      bierman::filters::UDFilter::update_scalar(ud, H, y - measure_scalar(ud.x), sigma * sigma);
    }
  });

  bench("SRIF-HH", [&] {
    Eigen::MatrixXd F0 = P0.ldlt().solve(Eigen::MatrixXd::Identity(n, n));
    bierman::filters::SRIFState s{F0.llt().matrixU(), F0.llt().matrixU() * x0};
    Eigen::MatrixXd Hm(1, n);
    Hm.row(0) = H;
    Eigen::MatrixXd sqrtW = (1.0 / sigma) * Eigen::MatrixXd::Identity(1, 1);

    for (int i = 0; i < iters; ++i) {
      Eigen::VectorXd x = bierman::linalg::solve_upper(s.R, s.z);
      Eigen::VectorXd y(1);
      y(0) = measure_scalar(x) + 0.01 * std::sin(0.01 * static_cast<double>(i));
      Eigen::VectorXd zlin = y - Eigen::VectorXd::Constant(1, measure_scalar(x)) + Hm * x;
      s = bierman::filters::SRIF::update_householder(s, Hm, zlin, sqrtW).state;
    }
  });

  bench("UKF", [&] {
    bierman::filters::UKFState ukf{x0, P0};
    bierman::filters::UTParams p{0.15, 0.0, 2.0};
    bierman::filters::UKFWorkspace ws;
    auto prop = [](const Eigen::VectorXd& x) { return x; };
    auto meas = [&](const Eigen::VectorXd& x) {
      Eigen::VectorXd y(1);
      y(0) = H.dot(x);
      return y;
    };
    Eigen::MatrixXd Q = 1e-4 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = sigma * sigma * Eigen::MatrixXd::Identity(1, 1);

    for (int i = 0; i < iters; ++i) {
      auto pred = bierman::filters::predict(ukf, Q, p, prop, &ws);
      Eigen::VectorXd y(1);
      y(0) = H.dot(pred.state.x) + 0.01 * std::sin(0.01 * static_cast<double>(i));
      ukf = bierman::filters::update(pred, y, R, meas, &ws);
    }
  });

  bench("SRUKF", [&] {
    bierman::filters::SRUKFState sr{x0, P0.llt().matrixL()};
    bierman::filters::UTParams p{0.15, 0.0, 2.0};
    bierman::filters::SRUKFWorkspace ws;
    bierman::filters::SRUKFDiagnostics diag;
    auto prop = [](const Eigen::VectorXd& x) { return x; };
    auto meas = [&](const Eigen::VectorXd& x) {
      Eigen::VectorXd y(1);
      y(0) = H.dot(x);
      return y;
    };
    Eigen::MatrixXd Q = 1e-4 * Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd R = sigma * sigma * Eigen::MatrixXd::Identity(1, 1);
    Eigen::MatrixXd Lq = Q.llt().matrixL();
    Eigen::MatrixXd Lr = R.llt().matrixL();

    for (int i = 0; i < iters; ++i) {
      auto pred = bierman::filters::SRUKF::predict(sr, Lq, p, prop, &ws, &diag);
      Eigen::VectorXd y(1);
      y(0) = H.dot(pred.state.x) + 0.01 * std::sin(0.01 * static_cast<double>(i));
      sr = bierman::filters::SRUKF::update(pred, y, Lr, meas, 0, &ws, &diag);
    }
  });

  return 0;
}
