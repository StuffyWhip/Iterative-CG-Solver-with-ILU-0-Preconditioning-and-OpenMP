#include "MatrixCRS.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <tuple>
#include <string>
#include <queue>
#include <algorithm>
#include <omp.h>

void MatrixCRS::readMatrixMarket(const std::string& fname) {
    std::ifstream in(fname);
    if (!in) throw std::runtime_error("Cannot open matrix file: "+fname);
    std::string line;
    std::getline(in,line);
    bool sym = line.find("symmetric")!=std::string::npos;
    while(std::getline(in,line) && !line.empty() && line[0]=='%');
    std::istringstream hdr(line);
    int M,N,L; hdr>>M>>N>>L;
    if(M!=N) throw std::runtime_error("Matrix not square");
    n=M;
    std::vector<std::tuple<int,int,double>> buf;
    buf.reserve(L*(sym?2:1));
    std::vector<int> rowc(n,0);
    int i,j; double v;
    for(int k=0;k<L;++k){
        in>>i>>j>>v; --i;--j;
        buf.emplace_back(i,j,v); rowc[i]++;
        if(sym && i!=j){ buf.emplace_back(j,i,v); rowc[j]++; }
    }
    rowPtr.assign(n+1,0);
    for(int r=0;r<n;++r) rowPtr[r+1]=rowPtr[r]+rowc[r];
    int nnz=rowPtr[n];
    colIdx.resize(nnz);
    vals.resize(nnz);
    std::vector<int> ptr=rowPtr;
    for(auto &t:buf){
        std::tie(i,j,v)=t;
        int p=ptr[i]++; colIdx[p]=j; vals[p]=v;
    }
}

void MatrixCRS::multiply(const std::vector<double>& x,
                         std::vector<double>&       y) const {
    if(y.size()!=size_t(n)) y.assign(n,0.0);
    else
#pragma omp parallel for schedule(static)
        for(int i=0;i<n;++i) y[i]=0.0;
#pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i){
        double s=0;
        for(int k=rowPtr[i];k<rowPtr[i+1];++k)
            s+=vals[k]*x[colIdx[k]];
        y[i]=s;
    }
}

std::vector<int> MatrixCRS::computeRCM() const {
    std::vector<std::vector<int>> adj(n);
    for(int i=0;i<n;++i)
        for(int k=rowPtr[i];k<rowPtr[i+1];++k){
            int j=colIdx[k];
            if(j!=i) adj[i].push_back(j);
        }
    std::vector<bool> seen(n,false);
    std::vector<int> perm; perm.reserve(n);
    std::vector<int> deg(n);
    for(int i=0;i<n;++i) deg[i]=adj[i].size();
    for(int start=0;start<n;++start){
        if(seen[start]) continue;
        int mn=start;
        for(int i=start;i<n;++i) if(!seen[i] && deg[i]<deg[mn]) mn=i;
        std::queue<int> q;
        seen[mn]=true;
        q.push(mn);
        while(!q.empty()){
            int u=q.front(); q.pop();
            perm.push_back(u);
            auto &nb=adj[u];
            std::sort(nb.begin(),nb.end(),[&](int a,int b){return deg[a]<deg[b];});
            for(int v:nb){
                if(!seen[v]){
                    seen[v]=true;
                    q.push(v);
                }
            }
        }
    }
    std::reverse(perm.begin(),perm.end());
    return perm;
}

void MatrixCRS::permute(const std::vector<int>& perm) {
    int N = n;
    std::vector<int> inv(N);
    for(int i=0;i<N;++i) inv[perm[i]]=i;
    std::vector<std::tuple<int,int,double>> buf;
    buf.reserve(vals.size());
    for(int i=0;i<N;++i){
        for(int k=rowPtr[i];k<rowPtr[i+1];++k){
            int j=colIdx[k];
            buf.emplace_back(inv[i],inv[j],vals[k]);
        }
    }
    rowPtr.assign(N+1,0);
    for(auto &t:buf) rowPtr[std::get<0>(t)+1]++;
    for(int i=1;i<=N;++i) rowPtr[i]+=rowPtr[i-1];
    int nnz=rowPtr[N];
    colIdx.assign(nnz,0);
    vals.assign(nnz,0.0);
    std::vector<int> ptr=rowPtr;
    for(auto &t:buf){
        int i0,j0; double vv;
        std::tie(i0,j0,vv)=t;
        int p=ptr[i0]++; colIdx[p]=j0; vals[p]=vv;
    }
}
