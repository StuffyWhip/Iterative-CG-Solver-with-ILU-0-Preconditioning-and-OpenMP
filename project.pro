TEMPLATE = app
CONFIG += console release c++17 openmp
QMAKE_CXXFLAGS += -std=c++17 -O3 -march=native -ffast-math -fopenmp
QMAKE_LFLAGS   += -fopenmp

SOURCES += \
    MatrixCRS.cpp \
    PreconditionerILU0.cpp \
    VectorUtils.cpp \
    ConjugateGradientSolver.cpp \
    main.cpp
HEADERS += \
    MatrixCRS.hpp \
    PreconditionerILU0.hpp \
    VectorUtils.hpp \
    ConjugateGradientSolver.hpp
