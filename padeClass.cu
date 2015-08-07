/*
padeClass.cu
Class for running the Pade exponentiation algorithm.

Nicolas Sawaya
2013
*/





template <typename tMat, typename tState, typename tReal>
class padeClass {

    public:
        padeClass();
        padeClass(systemClass<tMat,tState,tReal> *sysObjPtr,
            lanczosClass<tMat,tState,tReal> *lanObjPtr, tMat *d_expCoeff);


        bool doPade();
        bool doScalingStep();
        bool doUnscaling();

        bool doMatMult(
            denseMatrixClass<tMat> matResult,
            tMat* alpha,
            denseMatrixClass<tMat> matA,
            denseMatrixClass<tMat> matB,
            tMat* beta
            );
        bool doMatAdd(
            denseMatrixClass<tMat> matC,
            tMat* alpha,
            denseMatrixClass<tMat> matA,
            tMat* beta,
            denseMatrixClass<tMat> matB
            );
        bool runGesv();

        void cleanup();

        tReal getKrylovError();


        systemClass<tMat,tState,tReal> *sysObjPtr;
        lanczosClass<tMat,tState,tReal> *lanObjPtr;

        //int m;
        int lenMat; //Length of matrix to exponentiate. This is larger than m.
        int p; //order of the Pade expansion
        int s_scaling; //Scaling by 2^s
        int scaleMax; //i.e. norm cannot be greater than ~2^50

        //Coefficient in exponent. Usually set to (-i*dt/hbar)
        tMat* d_expCoeff;

        //Timestep
        //tReal* d_dt;
        //Using timestep from sysObj instead

        //rowSums used to determine matrix's infinity-norm
        thrust::device_vector<tReal> d_rowSums;
        thrust::host_vector<tReal> h_rowSums;

        //Powers of 2, and c_k vals used in algorithm.
        thrust::device_vector<tReal> d_2pow_neg_s;
        thrust::device_vector<tReal> d_2pow_s;
        thrust::device_vector<tMat> d_ckVals;
        thrust::host_vector<tReal> h_2pow_neg_s;
        thrust::host_vector<tReal> h_2pow_s;
        thrust::host_vector<tMat> h_ckVals;
        int *d_ipivCula; //Used by culaDgesv() function. Dimension is side of matrix.


        //Result of calculation
        denseMatrixClass<tMat> matExp_tT; //Result of Calculation


        //Intermediate matrices
        denseMatrixClass<tMat> mat2;  //matrix^2
        denseMatrixClass<tMat> mat4;  //matrix^4
        denseMatrixClass<tMat> mat6;  //matrix^6
        denseMatrixClass<tMat> matI;     //identity
        denseMatrixClass<tMat> matNum;   //'numerator'
        denseMatrixClass<tMat> matDen;   //'denominator'
        denseMatrixClass<tMat> tempMat1;
        denseMatrixClass<tMat> tempMat2;
        denseMatrixClass<tMat> tempMat3;


};


//Constructor
template <typename tMat, typename tState, typename tReal>
padeClass<tMat,tState,tReal>::
padeClass(systemClass<tMat,tState,tReal> *sysObjPtr,
    lanczosClass<tMat,tState,tReal> *lanObjPtr, tMat *d_expCoeff) {


    this->sysObjPtr = sysObjPtr;
    this->lanObjPtr = lanObjPtr;
    //this->m = lanObjPtr->m;
    this->lenMat = lanObjPtr->lenMatT;

    //6 is good choice for p based on error analysis in EXPOKIT eq(4)
    //Newer papers say p=13 w/ diff infNorm threshold.
    this->p = 6;

    //Exponent coefficient
    this->d_expCoeff = d_expCoeff;


    //Allocate for matrices
    this->matExp_tT.allocateOnHostAndDevice(lenMat,lenMat);
    this->mat2.allocateOnHostAndDevice(lenMat,lenMat);
    this->mat4.allocateOnHostAndDevice(lenMat,lenMat);
    this->mat6.allocateOnHostAndDevice(lenMat,lenMat);
    this->matI.allocateOnHostAndDevice(lenMat,lenMat);
    this->matNum.allocateOnHostAndDevice(lenMat,lenMat);
    this->matDen.allocateOnHostAndDevice(lenMat,lenMat);
    this->tempMat1.allocateOnHostAndDevice(lenMat,lenMat);
    this->tempMat2.allocateOnHostAndDevice(lenMat,lenMat);
    this->tempMat3.allocateOnHostAndDevice(lenMat,lenMat);


    //Populate Identity (don't forget real and imaginary zeros)
    tMat complexOne, complexZero;
    complexOne.x = 1.; complexOne.y = 0.;
    complexZero.x = 0.; complexZero.y = 0.;
    for(int row=0;row<lenMat;row++){
        for(int col=0;col<lenMat;col++){
            if(row==col) {
                this->matI.setHostElement(row,col,complexOne);
            } else {
                this->matI.setHostElement(row,col,complexZero);
            }
        }
    }
    //Copy to device
    this->matI.copyThisHostToThisDevice();



    //Allocate for vectors, set all to zero
    this->scaleMax = 50;
    this->h_rowSums.resize(lenMat);
    this->h_2pow_neg_s.resize(scaleMax);
    this->h_2pow_s.resize(scaleMax);
    this->h_ckVals.resize(p+1);
    //Set all imaginary to zero
    int i,k;
    //for(i=0;i<h_rowSums.size();i++) h_rowSums[i].y = 0;
    //for(i=0;i<h_2pow_neg_s.size();i++) h_2pow_neg_s[i].y = 0;
    //for(i=0;i<h_2pow_s.size();i++) h_2pow_s[i].y = 0;
    for(i=0;i<h_ckVals.size();i++) h_ckVals[i].y = 0;


    //Fill c_k values
    this->h_ckVals[0].x = 1;
    tReal flt_p = (tReal)p, flt_k;
    for(k=1;k<h_ckVals.size();k++) {
        flt_k = (tReal)k;
        h_ckVals[k].x = h_ckVals[k-1].x
            * (flt_p + 1. - flt_k)/( (2*flt_p + 1. - flt_k) * flt_k );
    }
    //Print c_k vals
    cout << "c_k vals" << endl;
    for(k=0;k<h_ckVals.size();k++) {
        cout << h_ckVals[k].x << "  ";
    }
    cout << endl;

    //Fill 2pow matrices
    for(k=0;k<h_2pow_s.size();k++) {
        h_2pow_s[k] = pow( 2., (float)k );
        h_2pow_neg_s[k] = pow( 2., -(float)k );
    }


    //Copy host to device (don't need to explicitly allocate)
    this->d_rowSums = this->h_rowSums;
    this->d_2pow_neg_s = this->h_2pow_neg_s;
    this->d_2pow_s = this->h_2pow_s;
    this->d_ckVals = this->h_ckVals;


    //Allocate for CULA
    cudaMalloc( (void**)&this->d_ipivCula, this->lenMat*sizeof(int) );

}









//Run Pade algorithm on matrix in Lanczos object
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
doPade() {

    cublasStatus_t cbStatus;
    

    //Scale the T-matrix by tau
    // cbStatus = cublasZdscal( //real multiplier complex vector
    //     sysObjPtr->cbHandle,
    //     this->lenMat*this->lenMat,    //elements in 'vector'
    //     &sysObjPtr->d_dt[0],    //multiplier
    //     lanObjPtr->matT.getDeviceColumnPtr(0),  //'vector' to scale
    //     1   //stride
    //     );
    // if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
    //     cout << "ERROR at cublasZdscal() in doPade().";
    //     cout << " Aborting." << endl;
    //     return false;
    // }

    // //Scale the T-matrix by i/hbar
    // cbStatus = cublasZscal( //both multiplier and vector complex
    //     sysObjPtr->cbHandle,
    //     this->lenMat*this->lenMat,    //elements in 'vector'
    //     &lanObjPtr->d_neg_inv_hbar_imag[0],    //multiplier
    //     lanObjPtr->matT.getDeviceColumnPtr(0),  //'vector to scale'
    //     1   //stride
    //     );
    // if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
    //     cout << "ERROR multiplying by inverse-hbar in doPade().";
    //     cout << " Aborting." << endl;
    //     return false;
    // }


    cbStatus = cublasZscal( //both multiplier and vector complex
        sysObjPtr->cbHandle,
        this->lenMat*this->lenMat,    //elements in 'vector'
        this->d_expCoeff,    //multiplier
        lanObjPtr->matT.getDeviceColumnPtr(0),  //'vector to scale'
        1   //stride
        );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR multiplying by inverse-hbar in doPade().";
        cout << " Aborting." << endl;
        return false;
    }

    //Print scaled T-matrix
    // lanObjPtr->matT.copyThisDeviceToThisHost();
    // cout << "T matrix times expCoeff:" << endl;
    // lanObjPtr->matT.printHostMat();


    //Do scaling step (matrix is multiplied by a negative power of two)
    if( ! this->doScalingStep() ) return false;

    // cout << "after scaling:" << endl;
    // this->lanObj.matT.copyThisDeviceToThisHost();
    // this->lanObj.matT.printHostMat();



    //tempMat1 = c1*I + c3*A^2
    //tempMat2 = tempMat1 + c5*A^4
    //num = A*tempMat2
    //
    //tempMat1 = c0*I + c2*A^2
    //tempMat2 = tempMat1 + c4*A^4
    //tempMat3 = tempMat2 + c6*A^6
    //den = num + tempMat3
    //trsm with alpha=2
    //exp() = 1 + trsm_result


    //Calculate mat^2
    if( !
        this->doMatMult(
            this->mat2,
            lanObjPtr->d_one,
            lanObjPtr->matT,
            lanObjPtr->matT,
            lanObjPtr->d_zero
        )
    ) return false;

    // cout << "mat2:" << endl;
    // this->mat2.copyThisDeviceToThisHost();
    // this->mat2.printHostMat();


    //Calculate mat^4
    if( !
        this->doMatMult(
            this->mat4,
            lanObjPtr->d_one,
            this->mat2,
            this->mat2,
            lanObjPtr->d_zero
        )
    ) return false;
    // cout << "mat4:" << endl;
    // this->mat4.copyThisDeviceToThisHost();
    // this->mat4.printHostMat();

    //Calculate mat^6
    if( !
        this->doMatMult(
            this->mat6,
            lanObjPtr->d_one,
            this->mat2,
            this->mat4,
            lanObjPtr->d_zero
        )
    ) return false;
    // cout << "mat6:" << endl;
    // this->mat6.copyThisDeviceToThisHost();
    // this->mat6.printHostMat();

    //tempMat1 = c1*I + c3*A^2
    if( !
        this->doMatAdd( //C = alpha*A + beta*B
            this->tempMat1,
            thrust::raw_pointer_cast(&this->d_ckVals[1]),
            this->matI,
            thrust::raw_pointer_cast(&this->d_ckVals[3]),
            this->mat2
        )
    ) return false;

    // cout << "tempMat1:" << endl;
    // this->tempMat1.copyThisDeviceToThisHost();
    // this->tempMat1.printHostMat();


    //tempMat2 = tempMat1 + c5*A^4
    if( !
        this->doMatAdd(
            this->tempMat2,
            lanObjPtr->d_one,
            this->tempMat1,
            thrust::raw_pointer_cast(&this->d_ckVals[5]),
            this->mat4
        )
    ) return false;
    // cout << "tempMat2:" << endl;
    // this->tempMat2.copyThisDeviceToThisHost();
    // this->tempMat2.printHostMat();

    //matNum = 1*T*tempMat2
    if( !
        this->doMatMult(
            this->matNum,
            lanObjPtr->d_one,
            lanObjPtr->matT,
            this->tempMat2,
            lanObjPtr->d_zero
        )
    ) return false;

    // cout << "matNum:" << endl;
    // this->matNum.copyThisDeviceToThisHost();
    // this->matNum.printHostMat();


    //tempMat1 = c0*I + c2*A^2
    if( !
        this->doMatAdd( //C = alpha*A + beta*B
            this->tempMat1,
            thrust::raw_pointer_cast(&this->d_ckVals[0]),
            this->matI,
            thrust::raw_pointer_cast(&this->d_ckVals[2]),
            this->mat2
        )
    ) return false;
    // cout << "tempMat1:" << endl;
    // this->tempMat1.copyThisDeviceToThisHost();
    // this->tempMat1.printHostMat();



    //tempMat2 = tempMat1 + c4*A^4
    if( !
        this->doMatAdd( //C = alpha*A + beta*B
            this->tempMat2,
            lanObjPtr->d_one,
            this->tempMat1,
            thrust::raw_pointer_cast(&this->d_ckVals[4]),
            this->mat4
        )
    ) return false;
    // cout << "tempMat2:" << endl;
    // this->tempMat2.copyThisDeviceToThisHost();
    // this->tempMat2.printHostMat();


    //tempMat3 = tempMat2 + c6*A^6
    if( !
        this->doMatAdd( 
            this->tempMat3,
            lanObjPtr->d_one,
            this->tempMat2,
            thrust::raw_pointer_cast(&this->d_ckVals[6]),
            this->mat6
        )
    ) return false;
    // cout << "tempMat3:" << endl;
    // this->tempMat3.copyThisDeviceToThisHost();
    // this->tempMat3.printHostMat();


    // cout << "matDen before anything:" << endl;
    // this->matDen.copyThisDeviceToThisHost();
    // this->matDen.printHostMat();
    //matDen = tempMat3 - matNum
    if( !
        this->doMatAdd( 
            this->matDen,
            lanObjPtr->d_one,
            this->tempMat3,
            lanObjPtr->d_negOne,
            this->matNum
        )
    ) return false;

    // cout << "matDen:" << endl;
    // this->matDen.copyThisDeviceToThisHost();
    // this->matDen.printHostMat();
    // cout << "tempMat3:" << endl;
    // this->tempMat3.copyThisDeviceToThisHost();
    // this->tempMat3.printHostMat();
    // cout << "matNum:" << endl;
    // this->matNum.copyThisDeviceToThisHost();
    // this->matNum.printHostMat();



    //Get numerator/denominator (using CULA's gesv)
    //ANSWER IS STORED IN this->matNum.
    if( ! this->runGesv() ) return false;


    // //Pointer for different places to store result
    // tMat *resultDest, *source;
    // if(this->s_scaling!=0)


    //EXP = I + 2*(Q\P)
    //= tempMat3 = matI + 2*num
    denseMatrixClass<tMat> destMat;
    if(this->s_scaling==0) {
        destMat = this->matExp_tT;
    } else {
        destMat = this->tempMat3;
    }
    if( !
        this->doMatAdd( 
            destMat,
            lanObjPtr->d_one,
            this->matI,
            lanObjPtr->d_two,
            this->matNum
        )
    ) return false;
    // cout << "tempMat3 (non-rescaled exponentiation):" << endl;
    // this->tempMat3.copyThisDeviceToThisHost();
    // this->tempMat3.printHostMat();



    //Scale back
    if( ! this->doUnscaling() ) return false;
    

    //Print result
    // cout << "RESULT exp(tau*T) = " << endl;
    // this->matExp_tT.copyThisDeviceToThisHost();
    // this->matExp_tT.printHostMat();

    return true;


}




//General matrix multiplication C = alpha*A*B + beta*C
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
doMatMult(
    denseMatrixClass<tMat> matC,
    tMat* alpha,
    denseMatrixClass<tMat> matA,
    denseMatrixClass<tMat> matB,
    tMat* beta
    ) {

    cublasStatus_t cbStatus;

    tMat* ptrC = matC.getDeviceColumnPtr(0);
    tMat* ptrA = matA.getDeviceColumnPtr(0);
    tMat* ptrB = matB.getDeviceColumnPtr(0);

    cbStatus = cublasZgemm( //C = alpha*A*B + beta*C
        this->sysObjPtr->cbHandle,
        CUBLAS_OP_N, //cublasOperation_t transa
        CUBLAS_OP_N, //cublasOperation_t transb
        this->lenMat, //m, number of rows of matrix A and C
        this->lenMat, //n, number of cols of matrix B and C
        this->lenMat, //k, number of cols of A and B
        alpha, //alpha
        ptrA, //A
        this->lenMat, //leading dimension of A
        ptrB, //B
        this->lenMat, //leading dimension of B
        beta, //beta
        ptrC, //C
        this->lenMat //leading dimension of C
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR in padeClass::doMatMult().";
        cout << " Error code " << cublasGetErrorString(cbStatus);
        cout << ". Aborting." << endl;
        return false;
    }

    return true;

}


//General matrix addition C = alpha*A + beta*B
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
doMatAdd(
    denseMatrixClass<tMat> matC,
    tMat* alpha,
    denseMatrixClass<tMat> matA,
    tMat* beta,
    denseMatrixClass<tMat> matB
    ) {


    cublasStatus_t cbStatus;

    tMat* ptrC = matC.getDeviceColumnPtr(0);
    tMat* ptrA = matA.getDeviceColumnPtr(0);
    tMat* ptrB = matB.getDeviceColumnPtr(0);

    cbStatus = cublasZgeam( //C = alpha*A + beta*B
        this->sysObjPtr->cbHandle,
        CUBLAS_OP_N, //op on A
        CUBLAS_OP_N, //op on B
        this->lenMat, //m rows
        this->lenMat, //n columns
        alpha, //alpha
        ptrA, //A
        this->lenMat, //leading dim of A
        beta, //beta
        ptrB, //B
        this->lenMat, //leading dim of B
        ptrC, //C
        this->lenMat //leading dim of C
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR in padeClass::doMatMult().";
        cout << " Error code " << cublasGetErrorString(cbStatus);
        cout << ". Aborting." << endl;
        return false;
    }

    return true;

}



//Scale the matrix after calculating the norm, etc.
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
doScalingStep() {


    //Compute sum of each row (remember it's in column-major format)
    calcRowSums<<< 1 , this->lenMat >>>(
        lanObjPtr->matT.getDeviceColumnPtr(0),
        thrust::raw_pointer_cast(&this->d_rowSums[0])
        );

    //Copy row maximums back to host
    thrust::copy( d_rowSums.begin(), d_rowSums.end(), h_rowSums.begin() );


    //Find maximum
    int i;
    tReal matInfNorm = 0.;
    for(i=0;i<h_rowSums.size();i++) {
        if(h_rowSums[i]>matInfNorm) {
            matInfNorm = h_rowSums[i];
        }
    }
    //cout << "matInfNorm = " << matInfNorm << endl;


    //Determine s to be used in 2^-s scaling
    int s=0;
    tReal nrm = matInfNorm;
    while(nrm>0.5) {
        nrm = 0.5 * nrm;
        s = s + 1;
        cout << "For scaling, s = " << s << endl;
    }
    if(s>=scaleMax) {
        cout << "ERROR. Scaling s>=scaleMax.";
        cout << " s = " << s << ", scaleMax = " << scaleMax;
        cout << "." << endl;
        return false;
    }

    //Set scaling for object
    this->s_scaling = s;

    //Scale tau*T by 2^(-s)
    cublasStatus_t cbStatus;
    cbStatus = cublasZdscal(
        sysObjPtr->cbHandle,
        this->lenMat*this->lenMat,    //elements in 'vector'
        thrust::raw_pointer_cast(&this->d_2pow_neg_s[s]),    //multiplier 2^(-s)
        lanObjPtr->matT.getDeviceColumnPtr(0),  //'vector' to scale
        1   //stride
        );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR at cublasZdscal() while scaling by 2^(-s).";
        cout << " Aborting." << endl;
        return false;
    }


    return true;




}





//"Unscaling"--continuous squaring an appropriate number of times
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
doUnscaling() {

    //Better strategy: use tempMat2 and tempMat3 except for the last step
    int s = this->s_scaling;

    int k;
    denseMatrixClass<tMat> resultDest, source;

    for( k=0; k<s; k++) {

        cout << "k = " << k << "." << endl;

        //Keep squaring, alternating between tempMat1 and tempMat2
        if(k % 2 == 0) {
            resultDest = this->tempMat2;
            source = this->tempMat1;
        } else {
            resultDest = this->tempMat1;
            source = this->tempMat2;
        }

        //If it's the last iteration then transfer to matExp_tT
        if(k==s-1) {
            resultDest = this->matExp_tT;
        }

        //If it's the first iteration then the source is tempMat3
        if(k==0) {
            source = this->tempMat3;
        }

        //Square the matrix
        if( ! doMatMult(
            resultDest,
            lanObjPtr->d_one,
            source,
            source,
            lanObjPtr->d_zero
            )
        ) return false;



    }

    return true;


}






//Run gesv to get numerator/denominator
template <typename tMat, typename tState, typename tReal>
bool padeClass<tMat,tState,tReal>::
runGesv() {

    culaStatus culaStat;

    tMat* ptrA = this->matDen.getDeviceColumnPtr(0);
    tMat* ptrB = this->matNum.getDeviceColumnPtr(0);


    //Solve matrix equation using CULA
    //B is num, A is den
    culaStat = culaDeviceZgesv( //Solve AX=B  Answer X ends up stored in B.
        this->lenMat, //(int) n, order of matrix A
        this->lenMat, //(int) nrhs, num rhs, i.e. num cols in B-mat
        (culaDoubleComplex*)ptrA, //(pointer) A, NxN
        this->lenMat, //(int) lda
        this->d_ipivCula, //(int pointer) ipiv, dimension N
        (culaDoubleComplex*)ptrB, //(pointer) B
        this->lenMat //(int) ldb
    );
    if(culaStat!=culaNoError) {
        cout << "ERROR runing culaDeviceZgesv. Aborting." << endl;
        return false;
    }




    return true;

}


//Return krylov error from local krylov object
template <typename tMat, typename tState, typename tReal>
tReal padeClass<tMat,tState,tReal>::
getKrylovError() {

    //This function uses Algorithm 3.2 from the expokit paper

    tReal err1, err2, errEstimate;
    tReal h_mp1_m = sqrt( pow(lanObjPtr->h_mp1_m.x,2) + pow(lanObjPtr->h_mp1_m.y,2) );
    tMat complexNum;
    int m = lanObjPtr->m;

    //Copy back part of matExp_tT
    this->matExp_tT.copyDeviceColsToHostColsMultiple( m , 2 );

    //Compute err1
    complexNum = this->matExp_tT.getHostElement(m-1,m);
    err1 = h_mp1_m * sqrt( pow(complexNum.x,2) + pow(complexNum.y,2) );

    //Computer err2
    complexNum = this->matExp_tT.getHostElement(m-1,m+1);
    err2 = h_mp1_m * lanObjPtr->norm2_A_v_mp1 * sqrt( pow(complexNum.x,2) + pow(complexNum.y,2) );

    //Algorithm 3.2 from expokit paper
    if(err1>err2) {
        errEstimate = err2 / (1 - err2/err1);
    } else { //err1<err2
        errEstimate = err1;
    }



    return errEstimate;

}


// //Return taylor-krylov error from local krylov object and from norm of state
// template <typename tMat, typename tState, typename tReal>
// tReal padeClass<tMat,tState,tReal>::
// getTaylorKrylovError( /* tReal norm */ ) {

//     tReal errEstimate;

//     //Error from Krylov part (this object is already storing the B-mat lanczos object)
//     tReal errKrylov = this->getKrylovError();

//     //Error estimate from norm


//     return errEstimate;

// }



//Cleanup
template <typename tMat, typename tState, typename tReal>
void padeClass<tMat,tState,tReal>::
cleanup() {

}





































