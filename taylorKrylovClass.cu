/*
taylorKrylovClass.cu
Class for Taylor-Krylov method.

Nicolas Sawaya
September 2013
*/



template <typename tMat, typename tState, typename tReal>
class taylorKrylovClass {

    public:
        taylorKrylovClass();
        taylorKrylovClass(
            systemClass<tMat,tState,tReal> *sysObjPtr,
            matCsrClass<tMat> *d_AmatCsr,
            string matBFileName,
            int p,
            int m_forBmat,
            tMat h_expCoeff,
            tMat *d_expCoeff
        );
        //void set_m(int m);


        //Object pointer (avoid using this altogether)
        systemClass<tMat,tState,tReal> *sysObjPtr;

        //Objects (not pointers)
        lanczosClass<tMat,tState,tReal> *lanObjBPtr;    //for B-matrix
        padeClass<tMat,tState,tReal> *padeObjBPtr;      //for B-matrix

        //Pointer to Hamiltonian


        bool setupProblem(); //don't know if necessary, but probably is

        bool propagateWithTaylorKrylov(tMat* ptrInVec, tMat* ptrOutVec);
        //bool doTaylorPart();
        //bool doKrylovPart();

        tReal getTaylorKrylovError(tReal vecNorm);

        bool sparseMatDenseVecMult(matCsrClass<tMat> d_csrMat, tMat* inVec, tMat* resultVec);
        bool do_Apowk_times_input(int k);


        void cleanup();


        //Size of system is N, size of decomposed matrix is m
        int N;
        int m_forBmat; //Used only on Krylov part
        int p; //This is number of 
        //int lenMatT; //This will be m+2 because of error calculation routine


        //Matrix pointers
        matCsrClass<tMat> *d_AmatCsr;

        //Matrices in memory
        matCooClass<tMat> h_BmatCoo;
        matCooClass<tMat> d_BmatCoo;
        matCsrClass<tMat> h_BmatCsr;
        matCsrClass<tMat> d_BmatCsr; //<-- this is the one we're prepping for

        //Vectors for calculating Taylor series
        //Switching back and forth between the two 'taylor components'
        thrust::device_vector<tMat> d_taylorCoeffs;
        thrust::device_vector<tMat> d_taylorComponent1;
        thrust::device_vector<tMat> d_taylorComponent2;
        tMat* d_taylorCompRawPtr1;
        tMat* d_taylorCompRawPtr2;

        //Vectors holding both parts of calculation
        thrust::device_vector<tMat> d_taylorPartVec;
        thrust::device_vector<tMat> d_krylovPartVec;
        //PROBABLY MERGE THESE

        //Pointers to the in and out vectors
        tMat* vecInPtr;
        tMat* vecOutPtr;

        //Pointers to zero-val and one-val
        tMat* d_zeroComplex;
        tMat* d_oneComplex;

        //This is the matrix coefficient, that I'm calling 'tau'
        tMat h_expCoeff;
        tMat* d_expCoeff;


        //h_m+1,m in Expokit paper, equals beta[m]
        //tMat h_mp1_m;

        // ||A*v_{m+1}||_2, used in error analysis
        //tReal norm2_A_v_mp1;



        //Constants for real parts
        //typeReal *d_vecNorm, *d_invVecNorm;

        //Which state vector (in other words, which time step)
        //int vecInNum;


        //CONST MEMORY???
        //Constants used in matrix operations
        //tMat  h_zero,  h_one,  h_two,  h_negOne, h_imag;
        //tMat *d_zero, *d_one, *d_two, *d_negOne, *d_imag;
        //tMat h_hbar_inv_cm_fs, h_hbar_J_s, h_lightspeed, h_planck_J_s;
        //tMat h_neg_inv_hbar_imag, *d_neg_inv_hbar_imag;




};



//Default Constructor
template <typename tMat, typename tState, typename tReal>
taylorKrylovClass<tMat,tState,tReal>::
taylorKrylovClass() {

}


//Constructor
template <typename tMat, typename tState, typename tReal>
taylorKrylovClass<tMat,tState,tReal>::
taylorKrylovClass(
    systemClass<tMat,tState,tReal> *sysObjPtr,
    matCsrClass<tMat> *d_AmatCsr,
    string matBFileName,
    int p,
    int m_forBmat,
    tMat h_expCoeff,
    tMat *d_expCoeff
    ) {


    //Avoid using this altogether
    this->sysObjPtr = sysObjPtr;

    //Assign expCoeff
    this->h_expCoeff = h_expCoeff;
    this->d_expCoeff = d_expCoeff;

    //Point to the main matrix ("A")
    this->d_AmatCsr = d_AmatCsr;

    //Read in matBFilename
    if( ! this->h_BmatCoo.readInFile(matBFileName, false) ) {
        cout << "ERROR in taylorKrylovClass constructor, reading in file." << endl;
    }
    
    //Set up the B-matrix. Prob want to switch matrix type later.
    this->d_BmatCoo.setCusparseHandle(& sysObjPtr->csHandle);
    this->h_BmatCoo.setCusparseHandle(& sysObjPtr->csHandle);
    this->d_BmatCsr.setCusparseHandle(& sysObjPtr->csHandle);
    this->h_BmatCsr.setCusparseHandle(& sysObjPtr->csHandle);

    //Set up Hamiltonian on system
    this->d_BmatCoo.createOnDevice(this->h_BmatCoo);
    //Copy to csr on device
    this->d_BmatCsr.pointToCooAndConvert(this->d_BmatCoo);
    //Copy csr to host. Don't do this unless testing.
    //if( ! this->h_BmatCsr.createFromDeviceCsrMat(this->d_BmatCsr) ) return false;


    //Copy system parameters over
    this->N = sysObjPtr->N;
    this->p = p;
    this->m_forBmat = m_forBmat;


    //Initialize internal lanczos and pade objects
    this->lanObjBPtr = new lanczosClass<tMat,tState,tReal>(
            sysObjPtr, this->m_forBmat, &(this->d_BmatCsr), this->p );     //for B-matrix

    this->padeObjBPtr =  new padeClass<tMat,tState,tReal>(sysObjPtr, this->lanObjBPtr, this->d_expCoeff);      //for B-matrix

    //Initialize vectors
    d_taylorCoeffs.resize(this->p);
    d_taylorComponent1.resize(this->N);
    d_taylorComponent2.resize(this->N);
    d_taylorPartVec.resize(this->N);
    d_krylovPartVec.resize(this->N);

    //Have these pointers for convenience later on
    d_taylorCompRawPtr1 = thrust::raw_pointer_cast(&d_taylorComponent1[0]);
    d_taylorCompRawPtr2 = thrust::raw_pointer_cast(&d_taylorComponent2[0]);

    //Populate Taylor coefficients.
    //Have 1,tau,tau^2/2,...,tau^(p-1)/(p-1)!,[zero] so that you can have 0 and 1 in there
    thrust::host_vector<tMat> h_taylorCoeffs;
    h_taylorCoeffs.resize(this->p+1);
    for(int k=0;k<p;k++) {
        h_taylorCoeffs[k] = powComplex(h_expCoeff,k);
        h_taylorCoeffs[k].x /= helperFactorial(k);
        h_taylorCoeffs[k].y /= helperFactorial(k);
        cout << "h_taylorCoeffs["<<k<<"].{x,y} = " << h_taylorCoeffs[k].x << "," << h_taylorCoeffs[k].y << endl;
    }
    h_taylorCoeffs[p].x = 0.; h_taylorCoeffs[p].y = 0.;

    //Copy constants to device
    d_taylorCoeffs = h_taylorCoeffs;


    //Just point to the values in the taylor series for the zero and one coefficients
    this->d_zeroComplex = thrust::raw_pointer_cast(&this->d_taylorCoeffs[p]);
    this->d_oneComplex = thrust::raw_pointer_cast(&this->d_taylorCoeffs[0]);

}

//Function to call when propagating with Taylor-Krylov
template <typename tMat, typename tState, typename tReal>
bool taylorKrylovClass<tMat,tState,tReal>::
propagateWithTaylorKrylov(tMat* ptrInVec, tMat* ptrOutVec) {

    //Update vector pointers in object
    this->vecInPtr = ptrInVec;
    this->vecOutPtr = ptrOutVec;





    // ** Do Taylor part first (up to A^(p-1)*vecIn) **

    //For first calculation, do A*w
    if(!
    sparseMatDenseVecMult(  //result = Mat*source. Asynchronous.
        *(this->d_AmatCsr),          //mat
        this->vecInPtr,             //source
        d_taylorCompRawPtr1    //result
        )
    ) return false;

    //Copy vector over to the taylorSum vector
    cublasStatus_t cbStatus;
    cbStatus = cublasZcopy(
        this->sysObjPtr->cbHandle,
        this->N,     //vector length
        ptrInVec,    //source
        1,  //stride
        thrust::raw_pointer_cast(&d_taylorPartVec[0]),   //result
        1   //stride
    );
    if(cbStatus != CUBLAS_STATUS_SUCCESS) { 
        cout << "ERROR with cublasZcopy in propagateWithTaylorKrylov. ";
        cout << "Aborting." << endl;
        return false;
    }

    //Synchronize, because csrmv() is asynchronous
    cudaDeviceSynchronize();
    
    //Pointers to switch between
    tMat* ptrTaylorCompResult = this->d_taylorCompRawPtr2;
    tMat* ptrTaylorCompSource = this->d_taylorCompRawPtr1; //Corresponds to A*vec


    //Start looping
    for( int k=2; k <= (this->p); k++ ) {

        //Compute A^k with cusparse (asynchronous)
        if(!
        sparseMatDenseVecMult(  //result = Mat*source. Asynchronous.
            *(this->d_AmatCsr),     //mat
            ptrTaylorCompSource,    //source
            ptrTaylorCompResult     //result
            )
        ) return false;

        //Do taylorSum = taylorSum + coeff*A^(k-1)*w
        cublasZaxpy(    //y = y + alpha*x
            this->sysObjPtr->cbHandle,  //handle
            this->N,                    //vector length
            thrust::raw_pointer_cast(&this->d_taylorCoeffs[k-1]), //alpha
            ptrTaylorCompSource,     //x. This is A^(k-1).
            1,  //stride
            thrust::raw_pointer_cast(&d_taylorPartVec[0]),    //y. The taylorSum.
            1    //stride
        );

        //Synchronize, because cusparse command was asynchronous
        cudaDeviceSynchronize();

        //Update pointer being used
        if(ptrTaylorCompResult == this->d_taylorCompRawPtr1) {
            ptrTaylorCompResult = this->d_taylorCompRawPtr2;
            ptrTaylorCompSource = this->d_taylorCompRawPtr1;
        } else if(ptrTaylorCompResult == this->d_taylorCompRawPtr2) {
            ptrTaylorCompResult = this->d_taylorCompRawPtr1;
            ptrTaylorCompSource = this->d_taylorCompRawPtr2;
        } else {
            cout << "ERROR in propagateWithTaylorKrylov updating pointer!!" << endl;
            return false;
        }

    }


    //Calculate norm, which is used for error calculation



    //Copy back and print taylorpart to test
    // thrust::host_vector<tMat> hostVec;
    // hostVec.resize(d_taylorPartVec.size());
    // hostVec = d_taylorPartVec;
    // cout << "Taylor part: " << endl;
    // for(int elem=0 ; elem<hostVec.size() ; elem++) {
    //     cout << setw(14) << hostVec[elem].x << "  " << hostVec[elem].y;
    //     cout << " i ("<<elem<<")" << endl;
    // }


    //Check value of A^n*vec
    // cudaMemcpy(thrust::raw_pointer_cast(&hostVec[0]),ptrTaylorCompSource,
    //     this->N * sizeof(hostVec[0]),
    //     cudaMemcpyDeviceToHost);
    // cout << "A^n*vec: " << endl;
    // for(int elem=0 ; elem<hostVec.size() ; elem++) {
    //     cout << setw(14) << hostVec[elem].x << "  " << hostVec[elem].y;
    //     cout << " i ("<<elem<<")" << endl;
    // }





    // ** Do Krylov part **

    //Do lanczos decomposition
    lanObjBPtr->doLanczos(ptrTaylorCompSource, true /* = calcAndStoreNormAndScale */); //Using pointer for A^p*vec

    //Copy back and print norm of A^n*vec, to test
    // tMat normVal = lanObjBPtr->copyBackAndGetNorm();
    // cout << "Norm of A^n*vec = " << normVal.x << "  " << normVal.y << endl;


    //Exponentiate the augmented matrix
    padeObjBPtr->doPade();

    //Copy back and print matExp_tT to test
    // padeObjBPtr->matExp_tT.copyThisDeviceToThisHost();
    // cout << "matExp_tT matrix:" << endl;
    // padeObjBPtr->matExp_tT.printHostMat();

    //Get pointer for last column in matrix, which is equal to tau^p*phi_n()*e1
    tMat* ptrPhi_n = padeObjBPtr->matExp_tT.getDeviceColumnPtr(this->m_forBmat + this->p - 1);
    tMat* ptrQ = lanObjBPtr->matQ.getDeviceColumnPtr(1);

    //Do multiplication and put Krylov vector into place
    //taylorPart = Q_m * lastColumn
    cbStatus = cublasZgemv( //y = alpha*op(A)*x + beta*y
        sysObjPtr->cbHandle,
        CUBLAS_OP_N,
        this->N, //rows in mat
        this->m_forBmat, //cols in mat
        this->lanObjBPtr->d_norm, //** 2-norm of original vector **
        ptrQ, //The matrix (first column is just zeros)
        this->N, //leading dimension of matrix (number of elements per column)
        ptrPhi_n, //vector (first column of exp(tT) matrix)
        1, //stride
        lanObjBPtr->d_zero, //beta-multiplier zero.
        ptrOutVec, //result vector
        1 //stride
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR at cublasZgemv() in taylorKrylovClass::propagateSystem()." << endl;
        return false;
    }


    //Copy back and print krylov part to test
    // cudaMemcpy(thrust::raw_pointer_cast(&hostVec[0]),ptrOutVec,
    //     this->N * sizeof(hostVec[0]),
    //     cudaMemcpyDeviceToHost);
    // cout << "Krylov part: " << endl;
    // for(int elem=0 ; elem<hostVec.size() ; elem++) {
    //     cout << setw(14) << hostVec[elem].x << "  " << hostVec[elem].y;
    //     cout << " i ("<<elem<<")" << endl;
    // }


    //Add the taylor part (d_taylorPartVec) to the result
    cbStatus = cublasZaxpy(    //y = y + alpha*x
        this->sysObjPtr->cbHandle,  //handle
        this->N,                    //vector length
        lanObjBPtr->d_one,          //alpha
        thrust::raw_pointer_cast(&d_taylorPartVec[0]),      //x.
        1,  //stride
        ptrOutVec,    //y.
        1    //stride
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR adding taylor and krylov parts in taylorKrylovClass::propagateSystem()." << endl;
        return false;
    }


    return true;


}


//Single matrix multiplication
template <typename tMat, typename tState, typename tReal>
tReal taylorKrylovClass<tMat,tState,tReal>::
getTaylorKrylovError(tReal vecNorm) {

    tReal errEst;

    //Krylov part of error
    errEst = this->padeObjBPtr->getKrylovError();

    //Print to test
    cout << "Krylov part of error for this step: " << errEst << endl;
    cout << "Norm-estimate error for this step: " << abs(1.-vecNorm) << endl;

    //Norm error
    errEst = errEst + abs(1.-vecNorm);

    return errEst;

}


//Single matrix multiplication
template <typename tMat, typename tState, typename tReal>
bool taylorKrylovClass<tMat,tState,tReal>::
sparseMatDenseVecMult(matCsrClass<tMat> d_csrMat, tMat* inVec, tMat* resultVec) {

    //cout << "Beginning lanczosClass.sparseMatDenseVecMult()." << endl;

    //Creating a handle just to see if it will work
    // cusparseHandle_t Xcshandle=0;
    // cusparseCreate(&Xcshandle);
    // cusparseSetPointerMode(Xcshandle, CUSPARSE_POINTER_MODE_DEVICE);

    cusparseStatus_t csStatus;
    csStatus = cusparseZcsrmv( //y = alpha*op(A)*x + beta*y //ASYNCHRONOUS!

        sysObjPtr->csHandle,                    //handle
        //Xcshandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,   //operation
        d_csrMat.N,                         //Rows in mat A
        d_csrMat.N,                         //Columns in mat A
        d_csrMat.nnz,                       //nnz
        d_oneComplex,                              //constant
        d_csrMat.cuspMatDescr,              //matrix descriptor
        d_csrMat.csrValA,                   //values in csr matrix
        d_csrMat.csrRowPtrA,                   //csrRowPtrA
        d_csrMat.csrColIndA,                   //csrColIndA
        inVec,                              //x-vector
        d_zeroComplex,                             //the beta-value
        resultVec                           //y-vector (result)

    );
    if(csStatus!=CUSPARSE_STATUS_SUCCESS) {
        cout << "ERROR in sparseMatDenseVecMult(). ";
        cout << "Error was: " << cusparseGetErrorString(csStatus);
        cout << ". Aborting" << endl;
        cout << endl;
        return false;
    }

    //Synchronize, because csrmv() is asynchronous
    cudaDeviceSynchronize();

    return true;

}


// //Single matrix multiplication
// template <typename tMat, typename tState, typename tReal>
// bool taylorKrylovClass<tMat,tState,tReal>::
// sparseMatDenseVecMult(matCsrClass<tMat> d_csrMat, tMat* inVec, tMat* resultVec) {

//     //cout << "Beginning lanczosClass.sparseMatDenseVecMult()." << endl;

//     //Creating a handle just to see if it will work
//     // cusparseHandle_t Xcshandle=0;
//     // cusparseCreate(&Xcshandle);
//     // cusparseSetPointerMode(Xcshandle, CUSPARSE_POINTER_MODE_DEVICE);

//     cusparseStatus_t csStatus;
//     csStatus = cusparseZcsrmv( //y = alpha*op(A)*x + beta*y //ASYNCHRONOUS!

//         sysObjPtr->csHandle,                    //handle
//         //Xcshandle,
//         CUSPARSE_OPERATION_NON_TRANSPOSE,   //operation
//         d_csrMat.N,                         //Rows in mat A
//         d_csrMat.N,                         //Columns in mat A
//         d_csrMat.nnz,                       //nnz
//         d_one,                              //constant
//         d_csrMat.cuspMatDescr,              //matrix descriptor
//         d_csrMat.csrValA,                   //values in csr matrix
//         d_csrMat.csrRowPtrA,                   //csrRowPtrA
//         d_csrMat.csrColIndA,                   //csrColIndA
//         inVec,                              //x-vector
//         d_zero,                             //the beta-value
//         resultVec                           //y-vector (result)

//     );
//     if(csStatus!=CUSPARSE_STATUS_SUCCESS) {
//         cout << "ERROR in sparseMatDenseVecMult(). ";
//         cout << "Error was: " << cusparseGetErrorString(csStatus);
//         cout << ". Aborting" << endl;
//         cout << endl;
//         return false;
//     }

//     //Synchronize, because csrmv() is asynchronous
//     cudaDeviceSynchronize();

//     return true;

// }


































