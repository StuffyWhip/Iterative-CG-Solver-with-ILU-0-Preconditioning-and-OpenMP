#include "VectorUtils.hpp"
#include <cmath>
#include <omp.h>

double VectorUtils::dot(const std::vector<double>& a,
                        const std::vector<double>& b) {
    int n=a.size(); double s=0;
#pragma omp parallel for reduction(+:s) schedule(static)
    for(int i=0;i<n;++i) s+=a[i]*b[i];
    return s;
}
double VectorUtils::norm2(const std::vector<double>& a){
    return std::sqrt(dot(a,a));
}
void VectorUtils::axpy(double α,const std::vector<double>& x,
                       std::vector<double>& y){
    int n=x.size();
#pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i) y[i]+=α*x[i];
}
