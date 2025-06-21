#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <getopt.h>
#include <omp.h>
#include "MatrixCRS.hpp"
#include "PreconditionerILU0.hpp"
#include "ConjugateGradientSolver.hpp"

int main(int argc, char* argv[]) {
    std::string matFile, rhsFile;
    double eps       = 1e-6;
    int    maxIter   = 100000;
    int    threads   = 7;
    int    restart   = 50;
    int    noPrecond = 1;

    static struct option longOpts[] = {
        {"no-precond", no_argument,       0, 1},
        {"epsilon",    required_argument, 0,          'e'},
        {"max_iter",   required_argument, 0,          'i'},
        {"threads",    required_argument, 0,          't'},
        {"restart",    required_argument, 0,          'r'},
        {0,0,0,0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "e:i:t:r:", longOpts, nullptr)) != -1) {
        switch (opt) {
        case 0:  /* --no-precond */           break;
        case 'e': eps      = std::stod(optarg); break;
        case 'i': maxIter  = std::stoi(optarg); break;
        case 't': threads  = std::stoi(optarg); break;
        case 'r': restart  = std::stoi(optarg); break;
        }
    }

    if (optind >= argc) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [rhs.txt] [--no-precond]"
                  << " [--epsilon] [--max_iter] [--threads] [--restart]\n";
        return 1;
    }

    matFile = argv[optind++];
    if (optind < argc) rhsFile = argv[optind];
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, 0);

    MatrixCRS A;
    A.readMatrixMarket(matFile);

    // сжимаем ненулевые эл., уменьшая профиль матрицы и ширину полосы (RCM-Reverse Cuthill–McKee)
    auto perm = A.computeRCM();
    A.permute(perm);

    // добавляем сигму для утолщения диагонали и избежания разрыва (невозможность взять корень у 0 числа) ILU
    const double sigma = 1e-6;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < A.n; ++i) {
        for (int k = A.rowPtr[i]; k < A.rowPtr[i+1]; ++k) {
            if (A.colIdx[k] == i) {
                A.vals[k] += sigma;
                break;
            }
        }
    }

    // читаем правую часть или создаем её
    std::vector<double> b(A.n), x(A.n, 0.0);
    if (!rhsFile.empty()) {
        std::ifstream in(rhsFile);
        for (int i = 0; i < A.n && (in >> b[i]); ++i);
    } else {
        std::fill(b.begin(), b.end(), 1.0);
    }
    // переставляем правую часть по RCM иначе система будет несовместима
    std::vector<double> b2(A.n);
    for (int i = 0; i < A.n; ++i) b2[i] = b[perm[i]];
    b.swap(b2);

    // строим предобусловливатель
    PreconditionerILU0 M;
    M.build(A);  // соберёт ILU или Jacobi-fallback
    const PreconditionerILU0* pM = noPrecond ? nullptr : &M;

    // запуск решателя
    ConjugateGradientSolver solver;
    CGResult res = solver.solve(
        A, b, x, pM,
        eps, maxIter, restart
        );

    // обратная перестановка x
    std::vector<double> x0(A.n);
    for (int i = 0; i < A.n; ++i)
        x0[perm[i]] = x[i];

    std::cout << "Read matrix: n=" << A.n
              << ", nnz=" << A.vals.size() << "\n"
              << "Iterations: " << res.iterations << "\n"
              << "Rel. error: "  << res.relError  << "\n"
              << "Time(threads:" << threads
              << (noPrecond ? ", no-precond" : "")
              << "): " << res.timeSec << " s\n";
    for (int i = 0; i < std::min(A.n, 5); ++i)
        std::cout << "x[" << i << "]=" << x0[i] << "\n";

    return 0;
}
