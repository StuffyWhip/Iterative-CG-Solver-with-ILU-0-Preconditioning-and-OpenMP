#pragma once
#include <vector>
#include "MatrixCRS.hpp"
#include "PreconditionerILU0.hpp"

struct CGResult {
    int    iterations;
    double relError;
    double timeSec;
};

class ConjugateGradientSolver {
public:
    /// @param M           — указатель на предобусловливатель; nullptr → identity
    CGResult solve(
        const MatrixCRS&           A,
        const std::vector<double>& b,
        std::vector<double>&       x,
        const PreconditionerILU0*  M,
        double                     epsilon,
        int                        maxIter,
        int                        restart
        );
};
