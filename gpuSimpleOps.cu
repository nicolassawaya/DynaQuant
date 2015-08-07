/*
gpuSimpleOps.cu
Simple operations that run on the GPU

Nicolas Sawaya
April 2013

*/



//Intended for use with only one element at a time, i.e. <<<1,1>>>
//Used to invert betas and alphas in lanczos algorithm

__global__ void negateSingle(typeMat* destArr, typeMat* sourceArr, int ind) {
    destArr[ind].x = - sourceArr[ind].x;
    destArr[ind].y = - sourceArr[ind].y;

}

__global__ void invertSingle(typeMat* destArr, typeMat* sourceArr, int ind) {
    //destArr[ind] = 1./sourceArr[ind];

    typeReal x = sourceArr[ind].x;
    typeReal y = sourceArr[ind].y;
    typeReal denom = x*x + y*y;

    destArr[ind].x = x / denom;
    destArr[ind].y = -y / denom;


    //Inverse of a complex number (a+bi) is (a-bi)/(a^2+b^2)
}

//Overloaded function
__global__ void invertSingle(typeReal* destArr, typeReal* sourceArr, int ind) {
    destArr[ind] = 1./sourceArr[ind];

}


/*
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}
int main()
{
...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
*/

