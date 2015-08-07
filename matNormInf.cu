/*
matNormInf.cu
Finds infinity-norm of a matrix

April 2013
Nicolas Sawaya
*/


//For matrix infinity-norm, sum all rows
//Remember this is in column format, so sum 
//Btw with lanczos this can be made much less complicated
__global__ void matNormInf(double* mat, double* result) {
    
    int row = threadIdx.x;
    int m = blockDim.x;
    double rowSum = 0.;
    int i;
    
    for(i=0;i<m;i++){
        //Matrix is in column-major format
        rowSum = rowSum + mat[i*m + row];
    }
    
    result[row] = rowSum;
    
}


