

#include <iostream>
//#include "cula.h"
#include <cula_lapack_device.h>


int main() {
    
    culaStatus culaStat;
    
    culaStat = culaInitialize();
    
    
    //Want to use culaDeviceDgesv
    culaStat = culaDgesv( //Solve AX=B
        , //(int) n, order of matrix A
        , //(int) nrhs, num rhs, i.e. num cols in B-mat
        , //(pointer) A, NxN
        , //(int) lda
        , //(int pointer) ipiv, dimension N
        , //(pointer) B
        , //(int) ldb
        
        
    );
    //B --> X as a result
    
    
    
    
    culaShutdown();
    
}

// nvcc culaTest.cu -lcula_core -lcula_lapack -lcublas -lcudart -I/usr/local/cula/include -L/usr/local/cula/lib64

