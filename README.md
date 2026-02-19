# bierman (C++20)

C++20 port of core algorithms from `motoq/bierman` (MPL-2.0) for factorization-based sequential estimation.

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
  - SRUKF (square-root form with robust fallback)
- Models
  - static range-only 3D box measurement model
  - dynamic ballistic-room style model (truth with drag, filter constant-velocity)
- Apps
  - `app_static_compare`
  - `app_dynamic_compare`
  - `app_ukf_srukf_bias`
- Tests
  - UD, QR, Cholesky update/downdate, Kalman Joseph
  - linear filter equivalence test
  - UKF/SRUKF consistency test

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Dependencies are pulled via `CPM.cmake`:
- Eigen 3.4.0
- spdlog v1.14.1

## Run apps

```bash
./build/app_static_compare --seed 7 --steps 100 --sigma_range 0.25
./build/app_dynamic_compare --seed 11 --dt 0.1 --tf 20 --sigma_range 0.2
./build/app_ukf_srukf_bias --seed 19 --dt 0.1 --tf 15 --sigma_range 0.3
```

Each app writes CSV outputs in `output_*` folders.

## Conventions

- Covariance square roots are lower triangular: `P = L L^T`
- UD form uses `P = U D U^T` with unit-upper `U`
- SRIF stores upper triangular `R` and RHS `z` such that `R x = z`

## Attribution

This repository ports algorithms and structure from:
- Upstream project: `https://github.com/motoq/bierman`
- Reference text: Gerald J. Bierman, *Factorization Methods for Discrete Sequential Estimation* (1977)

Original and ported code are distributed under MPL-2.0.
