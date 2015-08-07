/*
gpuLanczosHelper.cu
Containts GPU functions (which I cannot place in lanczosClass object)

Nicolas Sawaya
2013
*/






__global__ void populateMatT( typeMat* T, typeMat* alphas, typeMat* betas, int m ) {


    //NOTE:
    //blockSize - m = maximum phi_n number. Default should be 2.
    //Matrix is filled according to Theorem 1 in EXPOKIT paper.


    //Should be m blocks, m threads
    int matSize = blockDim.x;
    int i = blockIdx.x;
    int j = threadIdx.x;
    //int ind = j*m + i;
    int ind = j*(matSize) + i;


    T[ind].x = 0.;
    T[ind].y = 0.;

    //Conditionals very bad for these though.
    //Should probably use CUDAstreams instead.
    //See what Profiler says.
    if( i<m && j<m ) { //Main part of matrix with alphas and betas

        if(j==i) {
            T[ind].x = alphas[i+1].x;
            T[ind].y = alphas[i+1].y;
        } else if(j==i+1) {
            T[ind].x = betas[i+1].x;
            T[ind].y = betas[i+1].y;
        } else if(i==j+1) {
            T[ind].x = betas[j+1].x;
            T[ind].y = betas[j+1].y;
        }

    } else { //Augmented part of matrix, because calculating phi_n quantities too

        if( i==0 && j==m ) { //Corresponding to filled element in 'e1'
            T[ind].x = 1.;
        } else if( i>=m && j>=m && i==j-1 ) { //Corresponding to the ones in the lower-right section of the matrix
            T[ind].x = 1.;
        }

    }


}

























