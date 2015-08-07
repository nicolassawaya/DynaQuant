/*
gpuPadeHelper.cu
GPU functions to be used by the Pade object.

Nicolas Sawaya
2013
*/





//For matrix infinity-norm, sum all rows
//Remember this is in column format, so sum
//Btw with lanczos this can be made much less complicated
__global__ void calcRowSums(typeMat* mat, typeReal* result) {

    int row = threadIdx.x;
    int m = blockDim.x;
    typeReal rowSum = 0.;
    int i;

    for(i=0;i<m;i++){
        //Matrix is in column-major format
        rowSum = rowSum + abs(mat[i*m + row].x);
    }

    result[row] = rowSum;

}














