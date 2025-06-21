#include "PreconditionerILU0.hpp"
#include <cmath>
#include <stdexcept>
#include <omp.h>

void PreconditionerILU0::build(const MatrixCRS& A) {
    n = A.n;
    rowPtr = A.rowPtr;
    colIdx = A.colIdx;
    int nnz = A.vals.size();
    L.assign(nnz, 0.0);
    U.assign(nnz, 0.0);

    for(int k=0;k<nnz;++k) U[k] = A.vals[k];

    for(int i=0;i<n;++i){
        double diagU = 0.0;
        for(int kk = rowPtr[i]; kk < rowPtr[i+1]; ++kk){
            int j = colIdx[kk];
            if(j < i){
                double Ujj = 0.0;
                for(int k2 = rowPtr[j]; k2<rowPtr[j+1]; ++k2)
                    if(colIdx[k2]==j){ Ujj = U[k2]; break; }
                if(std::abs(Ujj)<1e-14) throw std::runtime_error("ILU0 breakdown");
                L[kk] = U[kk] / Ujj;
                for(int k2 = rowPtr[j]; k2<rowPtr[j+1]; ++k2){
                    int c = colIdx[k2];
                    for(int k3=rowPtr[i]; k3<rowPtr[i+1]; ++k3){
                        if(colIdx[k3]==c){
                            U[k3] -= L[kk] * U[k2];
                            break;
                        }
                    }
                }
            } else if(j==i){
                diagU = U[kk];
                U[kk] = diagU;
            }
        }
    }
}

void PreconditionerILU0::apply(const std::vector<double>& r,
                               std::vector<double>&       z) const {
    int n = r.size();
    std::vector<double> y(n);
    for(int i=0;i<n;++i){
        double s=0;
        for(int k=rowPtr[i]; k<rowPtr[i+1]; ++k){
            int j=colIdx[k];
            if(j<i) s+=L[k]*y[j];
        }
        y[i] = (r[i]-s);  // Lii=1
    }
    // решаем U z = y  (U верхний треугольник)
    z.assign(n,0.0);
    for(int i=n-1;i>=0;--i){
        double s=0, uii=0;
        for(int k=rowPtr[i]; k<rowPtr[i+1]; ++k){
            int j=colIdx[k];
            if(j==i) uii=U[k];
            else if(j>i) s+=U[k]*z[j];
        }
        z[i] = (y[i]-s)/uii;
    }
}
