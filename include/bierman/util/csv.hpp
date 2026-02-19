#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace bierman::util {

inline void write_csv_header(std::ofstream& os, const std::vector<std::string>& cols) {
  for (size_t i = 0; i < cols.size(); ++i) {
    os << cols[i];
    if (i + 1 < cols.size()) {
      os << ',';
    }
  }
  os << '\n';
}

inline void write_csv_row(std::ofstream& os, const std::vector<double>& vals) {
  for (size_t i = 0; i < vals.size(); ++i) {
    os << vals[i];
    if (i + 1 < vals.size()) {
      os << ',';
    }
  }
  os << '\n';
}

inline void write_csv_row(std::ofstream& os, const std::string& key, double val) {
  os << key << ',' << val << '\n';
}

inline void write_csv_row(std::ofstream& os, double t, const Eigen::Ref<const Eigen::VectorXd>& x) {
  os << t;
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    os << ',' << x(i);
  }
  os << '\n';
}

}  // namespace bierman::util
