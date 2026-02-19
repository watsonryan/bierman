# bierman (C++20)

C++20 port of algorithms from `motoq/bierman` (MPL-2.0) for factorization-based sequential estimation.

## Implemented

- Linalg primitives
  - triangular solves
  - UD factorization and reconstruction
  - QR (Householder + MGS)
  - rank-one Cholesky update/downdate (lower)
- Filters
  - Joseph-form Kalman update (scalar sequential)
  - Potter square-root covariance update (scalar sequential)
  - UD filter update + prediction with diagonal process noise
  - SRIF update (Householder, QR/MGS) + SRIF prediction
  - Schmidt/consider Kalman update
  - UKF (augmented-state form)
  - SRUKF (square-root form with robust fallback + fallback diagnostics)
- Models
  - static range-only 3D box measurement model
  - dynamic ballistic-room style model (truth with drag, filter constant-velocity)
- Apps
  - `app_static_compare`
  - `app_dynamic_compare`
  - `app_ukf_srukf_bias`
- Tests
  - UD, QR, Cholesky update/downdate, Kalman Joseph
  - multi-step linear filter equivalence test
  - UKF/SRUKF consistency test
  - SRUKF stress test with fallback thresholds
  - deterministic golden regression metrics test

## API Boundary

- Public headers are under `include/bierman/...`.
- Internal/non-stable helpers live under `include/bierman/internal/...`.
- Code outside this repository should not include `bierman/internal/*` directly.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Dependencies are resolved via `CPM.cmake`:
- Eigen 5.0.1
- spdlog v1.17.0
- CLI11 v2.4.2

If dependencies are installed system-wide, fetches can be disabled:

```bash
cmake -S . -B build -DBIERMAN_FETCH_DEPS=OFF
```

## Run apps

```bash
./build/app_static_compare --seed 7 --steps 100 --sigma_range 0.25 --outdir output_static
./build/app_dynamic_compare --seed 11 --dt 0.1 --tf 20 --sigma_range 0.2 --outdir output_dynamic
./build/app_ukf_srukf_bias --seed 19 --dt 0.1 --tf 15 --sigma_range 0.3 --outdir output_ukf_srukf_bias
```

All apps support:
- `--outdir <path>`
- `--quiet`
- `--config <file>` (CLI11 INI/TOML config parsing)

## Conventions

- Covariance square roots are lower triangular: `P = L L^T`
- UD form uses `P = U D U^T` with unit-upper `U`
- SRIF stores upper triangular `R` and RHS `z` such that `R x = z`
- SRIF linearization convention in this repo:
  - `z_lin = y - h(x_ref) + H*x_ref`
  - update APIs return absolute-state solution from `R*x = z`

## Regression Baselines

Golden metrics are stored in:
- `tests/golden/metrics_baseline.txt`

`test_regression_metrics` compares deterministic runs against those values and fails on drift above tolerance.

## Attribution

This repository ports algorithms and structure from:
- Upstream project: `https://github.com/motoq/bierman`
- Reference text: Gerald J. Bierman, *Factorization Methods for Discrete Sequential Estimation* (1977)

Original and ported code are distributed under MPL-2.0.
