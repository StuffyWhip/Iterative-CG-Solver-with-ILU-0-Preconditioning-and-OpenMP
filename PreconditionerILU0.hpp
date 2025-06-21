#pragma once
#include "MatrixCRS.hpp"
#include <vector>

class PreconditionerILU0 {
public:
    int n;
    std::vector<int>    rowPtr, colIdx;
    std::vector<double> L, U;

    void build(const MatrixCRS& A);
    void apply(const std::vector<double>& r, std::vector<double>& z) const;
};
