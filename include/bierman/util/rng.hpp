#pragma once

#include <random>

namespace bierman::util {

class Rng {
 public:
  explicit Rng(unsigned int seed) : gen_(seed) {}

  double normal(double mean, double sigma) {
    std::normal_distribution<double> dist(mean, sigma);
    return dist(gen_);
  }

  double uniform(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(gen_);
  }

 private:
  std::mt19937 gen_;
};

}  // namespace bierman::util
