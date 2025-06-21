#include "ConjugateGradientSolver.hpp"
#include "VectorUtils.hpp"
#include <chrono>
#include <omp.h>

CGResult ConjugateGradientSolver::solve(
    const MatrixCRS&           A,
    const std::vector<double>& b,
    std::vector<double>&       x,
    const PreconditionerILU0*  M,
    double                     epsilon,
    int                        maxIter,
    int                        restart
    ) {
    int n = A.n;
    std::vector<double> r(n), z(n), p(n), Ap(n);

    // r = b - A*x
    A.multiply(x, Ap);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i)
        r[i] = b[i] - Ap[i];

    // z = M ? M->apply(r) : r
    if (M) M->apply(r, z);
    else   z = r;

    p = z;
    double rz_old = VectorUtils::dot(r, z);
    double bnorm  = VectorUtils::norm2(b);
    int    iter   = 0;
    double relRes = 1.0;

    auto t0 = std::chrono::high_resolution_clock::now();
    while (++iter <= maxIter) {
        A.multiply(p, Ap);
        double pAp   = VectorUtils::dot(p, Ap);
        double alpha = rz_old / pAp;
        VectorUtils::axpy( alpha, p, x);
        VectorUtils::axpy(-alpha, Ap, r);
        if (iter % restart == 0) {
            A.multiply(x, Ap);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i)
                r[i] = b[i] - Ap[i];
        }

        relRes = VectorUtils::norm2(r) / bnorm;
        if (relRes < epsilon) break;
        if (M) M->apply(r, z);
        else   z = r;
        double rz_new = VectorUtils::dot(r, z);
        double beta   = rz_new / rz_old;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            p[i] = z[i] + beta * p[i];

        rz_old = rz_new;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double timeSec = std::chrono::duration<double>(t1 - t0).count();

    return CGResult{iter, relRes, timeSec};
}
