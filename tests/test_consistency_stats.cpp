#include <spdlog/spdlog.h>

#include "bierman/util/stats.hpp"

int main() {
  bierman::util::ConsistencyAccumulator acc;
  for (int i = 0; i < 200; ++i) {
    acc.add_nis(1.0);
    acc.add_nees(3.0);
  }

  auto nis_gate = acc.nis_gate(1, 0.05);
  auto nees_gate = acc.nees_gate(3, 0.05);

  if (!nis_gate.in_gate || !nees_gate.in_gate) {
    spdlog::error("Expected nominal consistency means to lie in chi-square gates.");
    return 1;
  }

  return 0;
}
