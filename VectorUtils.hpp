#pragma once
#include <vector>

namespace VectorUtils {

double dot(const std::vector<double>& a,
           const std::vector<double>& b);

double norm2(const std::vector<double>& a);

void axpy(double alpha,
          const std::vector<double>& x,
          std::vector<double>&       y);

}
