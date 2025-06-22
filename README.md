# Iterative CG Solver with ILU(0) Preconditioning and OpenMP

**A C++ application** for solving large sparse symmetric positive-definite systems **Ax=b** using the Conjugate Gradient (CG) method with ILU(0) preconditioning (with Jacobi fallback) and full parallelization via OpenMP.

## Key Features

**Conjugate Gradient** (CG) with restarts

**ILU(0)** incomplete LU factorization without fill-in + Jacobi fallback

**Optional** preconditioner disable (--no-precond) for pure CG

**RCM permutation** (Reverse Cuthill–McKee) to reduce matrix bandwidth/profile

**Diagonal regularization** A<-A+σI

**Parallel execution** of all hot spots via OpenMP: 
  SpMV (CRS × vector) 
  dot products  
  AXPY (vector updates)
  forward/back substitution for ILU(0)  
  
**Configurable via command-line options**:  
  tolerance --epsilon 
  max iterations --max_iter 
  number of threads --threads 
  restart frequency --restart 
  disable preconditioning --no-precond

---
#  Build and Installation
bash

```
git clone https://github.com/User/Iterative-CG-Solver-with-ILU-0-Preconditioning-and-OpenMP.git
cd Iterative-CG-Solver-with-ILU-0-Preconditioning-and-OpenMP
qmake project.pro
make -j$(nproc)
```
To build without any preconditioning (pure CG):
```
qmake project.pro "DEFINES+=NO_PRECOND"
make -j$(nproc)
```
Site with matrixes https://sparse.tamu.edu/. 
1) Downdlaod "Matrix market" format for chosen matrix.
2) Put "name".mtx and rhs.txt into project_name/build/build_version/ folder
