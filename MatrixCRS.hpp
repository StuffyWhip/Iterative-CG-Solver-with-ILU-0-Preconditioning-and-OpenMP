#pragma once
#include <vector>
#include <string>

class MatrixCRS {
public:
    int n;
    std::vector<int>    rowPtr;
    std::vector<int>    colIdx;
    std::vector<double> vals;

    void readMatrixMarket(const std::string& fname);
    // y = A × x
    void multiply(const std::vector<double>& x, std::vector<double>& y) const;
    std::vector<int> computeRCM() const;
    void permute(const std::vector<int>& perm);
};
