/*
lanczosClass.cu
Class for running the lanczos algorithm

Nicolas Sawaya
2013
*/



template <typename tMat, typename tState, typename tReal>
class lanczosClass {

    public:
        lanczosClass();
        lanczosClass(systemClass<tMat,tState,tReal> *sysObjPtr,
            int m,
            matCsrClass<tMat> *d_matCsr,
            int matrixAugmentationLength=2);
        //void set_m(int m);


        bool doLanczos(tMat* ptrVecIn, bool calcAndStoreNormAndScale=false);
        bool setupProblem(); //Sets the first column of relevant matrices
        bool runStandardLanczos();
        bool createKrylovSpace();
        void lanczosDecompFromNaiveKrylov();

        bool find_norm2_A_v_mp1();
        //__global__ void populateMatT();

        bool sparseMatDenseVecMult(matCsrClass<tMat> d_csrMat, tMat* inVec, tMat* resultVec);
        bool do_Apowk_times_input(int k);

        tMat copyBackAndGetNorm();

        void cleanup();

        //Library objects
        //cusparseHandle_t csHandle;
        //cublasHandle_t cbHandle;

        //If 'Taylor-Krylov' method (B+A-B) should be used
        bool useTaylorKrylov;

        //Size of system is N, size of decomposed matrix is m
        int N;
        int m;
        int matrixAugLength;
        int lenMatT; //This will be m+2 because of error calculation routine

        //System object pointer
        systemClass<tMat,tState,tReal> *sysObjPtr;

        //Hamiltonian pointer
        matCsrClass<tMat> *d_matCsr;
        

        //Matrix objects to *point* to
        //matCsrClass<tMat> d_MatCsr;

        //State vector to *point* to
        //tState* d_inState;
        //tState* d_outState;

        //Vectors for the decomposition
        thrust::device_vector<tMat> d_alphas;
        thrust::device_vector<tMat> d_betas;
        thrust::device_vector<tMat> d_negAlphas;
        thrust::device_vector<tMat> d_negBetas;
        thrust::device_vector<tMat> d_invBetas;
        thrust::device_vector<tMat> d_w;

        //Matrices for the decomposition
        denseMatrixClass<tMat> matT;
        denseMatrixClass<tMat> matQ;
        // denseMatrixClass<tMat> matNaiveKrylov; //Simple Taylor series

        //h_m+1,m in Expokit paper, equals beta[m]
        tMat h_mp1_m;

        //Storing norm, if specified
        bool storingNormAndScaling;
        tMat h_norm, *d_norm, *d_invNorm;

        // ||A*v_{m+1}||_2, used in error analysis
        tReal norm2_A_v_mp1;



        //Constants for real parts
        //typeReal *d_vecNorm, *d_invVecNorm;

        //Which state vector (in other words, which time step)
        //int vecInNum;
        tMat* ptrVecIn;

        //CONST MEMORY???
        //Constants used in matrix operations
        tMat  h_zero,  h_one,  h_two,  h_negOne, h_imag;
        tMat *d_zero, *d_one, *d_two, *d_negOne, *d_imag;
        // tMat h_hbar_inv_cm_fs, h_hbar_J_s, h_lightspeed, h_planck_J_s;
        // tMat h_neg_inv_hbar_imag, *d_neg_inv_hbar_imag;




};


//Default Constructor
template <typename tMat, typename tState, typename tReal>
lanczosClass<tMat,tState,tReal>::
lanczosClass() {

}


//Constructor
template <typename tMat, typename tState, typename tReal>
lanczosClass<tMat,tState,tReal>::
lanczosClass(systemClass<tMat,tState,tReal> *sysObjPtr, int m,
    matCsrClass<tMat> *d_matCsr,
    int matrixAugmentationLength/* =2 (set in def)*/) {

    cout << "Initializing lanczos object." << endl;



    //Point to the matrix to be exponentiated
    this->sysObjPtr = sysObjPtr;
    this->d_matCsr = d_matCsr;
    this->N = sysObjPtr->N;
    this->m = m;
    this->matrixAugLength = matrixAugmentationLength;
    this->lenMatT = this->m + this->matrixAugLength; //For error calculation, we increase the length by 2



    //CONST MEMORY??? Meh. These are one-at-a-time operations.
    //Constants used in matrix operations
    cudaMalloc( (void**)&d_zero, sizeof(tMat) );
    cudaMalloc( (void**)&d_one, sizeof(tMat) );
    cudaMalloc( (void**)&d_two, sizeof(tMat) );
    cudaMalloc( (void**)&d_negOne, sizeof(tMat) );
    cudaMalloc( (void**)&d_imag, sizeof(tMat) );

    //For norm calculation and storage
    h_norm.x = h_norm.y = 0.;
    cudaMalloc( (void**)&d_norm, sizeof(*d_norm) );
    cudaMalloc( (void**)&d_invNorm, sizeof(*d_invNorm) );
    cudaMemcpy( d_norm, &h_norm, sizeof(h_norm), cudaMemcpyHostToDevice); //so that y is set to zero

    //cudaMalloc( (void**)&d_neg_inv_hbar_imag, sizeof(tMat) );

    h_zero.x = 0.; h_zero.y = 0.;
    h_one.x = 1.; h_one.y = 0.;
    h_two.x = 2.; h_two.y = 0.;
    h_negOne.x = -1.; h_negOne.y = 0;
    h_imag.x = 0.; h_imag.y = 1;

    cudaError_t cuErr;
    cuErr = cudaMemcpy( d_zero, &h_zero, sizeof(tMat), cudaMemcpyHostToDevice );
    cuErr = cudaMemcpy( d_one, &h_one, sizeof(tMat), cudaMemcpyHostToDevice );
    cuErr = cudaMemcpy( d_two, &h_two, sizeof(tMat), cudaMemcpyHostToDevice );
    cuErr = cudaMemcpy( d_negOne, &h_negOne, sizeof(tMat), cudaMemcpyHostToDevice );
    cuErr = cudaMemcpy( d_imag, &h_imag, sizeof(tMat), cudaMemcpyHostToDevice );
    //cuErr = cudaMemcpy( d_neg_inv_hbar_imag, &h_neg_inv_hbar_imag, sizeof(tMat), cudaMemcpyHostToDevice );
    if(cuErr!=cudaSuccess) {
        cout << "ERROR copying constants to device in lanczosClass constructor." << endl;
        exit(1);
        return;
    }

    //Allocate for vectors
    this->d_alphas.resize(m+2);
    this->d_betas.resize(m+2);
    this->d_negAlphas.resize(m+2);
    this->d_negBetas.resize(m+2);
    this->d_invBetas.resize(m+2);
    this->d_w.resize(this->N);

    //Set all of betas and alphas to zero.
    //This is strictly necessary only for alpha_0, 
    //beta_0, and all imaginary parts in the betas.
    thrust::fill(this->d_betas.begin(), this->d_betas.end(), this->h_zero);
    thrust::fill(this->d_invBetas.begin(), this->d_invBetas.end(), this->h_zero);
    // cuErr = cudaMemcpy(
    //     thrust::raw_pointer_cast(&this->d_betas[0]),
    //     this->d_zero,
    //     sizeof(tMat),
    //     cudaMemcpyDeviceToDevice
    // );
    // cuErr = cudaMemcpy(
    //     thrust::raw_pointer_cast(&this->d_invBetas[0]),
    //     this->d_zero,
    //     sizeof(tMat),
    //     cudaMemcpyDeviceToDevice
    // );
    // if(cuErr!=cudaSuccess) {
    //     cout << "ERROR setting beta_0 or invBeta_0 lanczosClass constructor." << endl;
    //     return;
    // }


    //Allocate for matrices
    this->matT.allocateOnHostAndDevice(this->lenMatT,this->lenMatT);
    this->matQ.allocateOnHostAndDevice(N,m+2);
    // this->matNaiveKrylov.allocateOnHostAndDevice(N,m+1);

    //Put zeros in first column of matQ and matNaiveKrylov
    this->matQ.setColumnToZero(0);
    // this->matNaiveKrylov.setColumnToZero(0);

    //For now, Taylor-Krylov method not used
    this->useTaylorKrylov = false;

    //Libraries
    //this->csHandle = sysObj.csHandle;
    //this->cbHandle = sysObj.cbHandle;




}



//Run lanczos decomposition
template <typename tMat, typename tState, typename tReal>
bool lanczosClass<tMat,tState,tReal>::
doLanczos(tMat* ptrVecIn, bool calcAndStoreNormAndScale/*=false*/) {

    //cout << "Beginning doLanczos()." << endl;

    this->storingNormAndScaling = calcAndStoreNormAndScale;

    //Which state vector to work with (based on which timestep you're on)
    // this->vecInNum = vecInNum;
    this->ptrVecIn = ptrVecIn;

    //Set up initial vectors
    this->setupProblem();



    //cout << "Not using Taylor-Krylov method." << endl;

    //Set up initial vectors
    if(! this->setupProblem() ) return false;


    if(this->storingNormAndScaling) {

        //Try with some new variables
        //tMat* test_d_norm;
        //cudaMalloc( (void**)&test_d_norm, sizeof(*test_d_norm))

        //See if synchronization works
        //cudaDeviceSynchronize();

        //Calculate Norm
        cublasStatus_t cbStatus;
        cbStatus = 
        cublasDznrm2(
            this->sysObjPtr->cbHandle,                  //Handle
            this->N,                                    //vector length
            this->matQ.getDeviceColumnPtr(1),           //First vector has been copied here
            1,      //stride
            (double*)(this->d_norm) //result
        );
        if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
            cout << "ERROR using cublasDznrm2 in doLanczos. Error = ";
            cout << cublasGetErrorString(cbStatus) << "." << endl;
            //return false;
        }

        // cbStatus = cublasDznrm2(
        //     this->sysObjPtr->cbHandle,                      //Handle
        //     this->N,                                    //vector length
        //     thrust::raw_pointer_cast(&this->d_w[0]),    //vector
        //     1,      //stride
        //     (double*) thrust::raw_pointer_cast(&this->d_betas[j])
        // );

        //Copy back and prin d_norm
        // cudaDeviceSynchronize(); //Synchronize I think, because nrm2() is asynchronous
        // tMat tMatVal;
        // cudaMemcpy(&tMatVal, this->d_norm, sizeof(tMatVal), cudaMemcpyDeviceToHost);
        // cout << "d_norm = " << tMatVal.x << " " << tMatVal.y << " i" << endl;

        //Invert Norm
        invertSingle<<<1,1>>>(
            this->d_invNorm, //dest
            this->d_norm,    //source
            0 //index
        );

        //Scale
        //cbStatus = 
        cublasZscal(
            this->sysObjPtr->cbHandle,
            this->N,
            this->d_invNorm,
            matQ.getDeviceColumnPtr(1),
            1 //stride
        );


    }


    //Run algorithm
    if(! this->runStandardLanczos() ) return false;

    //Populate matT with alphas and betas
    //populateMatT<<< this->m , this->m >>>(
    populateMatT<<< this->lenMatT, this->lenMatT >>>(
        this->matT.getDeviceColumnPtr(0),
        thrust::raw_pointer_cast(&this->d_alphas[0]),
        thrust::raw_pointer_cast(&this->d_betas[0]),
        this->m
    );

    //Find ||A*v_{m+1}||_2, used in error analysis
    if(! this->find_norm2_A_v_mp1() ) return false;
    //Print it out
    // cout << "this->norm2_A_v_mp1 = " << this->norm2_A_v_mp1 << endl;

    //Copy back h_m+1,m
    cudaMemcpy(&this->h_mp1_m, thrust::raw_pointer_cast(&this->d_betas[m]), sizeof( (this->h_mp1_m) ), cudaMemcpyDeviceToHost );
    
    //Print h_m+1,m
    // cout << "this->h_mp1_m = " << this->h_mp1_m.x << " + " << this->h_mp1_m.y << " *i" << endl;


    // Copy back matT and print
    // matT.copyThisDeviceToThisHost();
    // cout << "T matrix:" << endl;
    // matT.printHostMat();

    // Copy back matQ and print
    // matQ.copyThisDeviceToThisHost();
    // cout << "Q matrix:" << endl;
    // matQ.printHostMat();


    


    return true;

}


//Set up initial vectors in relevant matrices, set up rest of problem
template <typename tMat, typename tState, typename tReal>
bool lanczosClass<tMat,tState,tReal>::
setupProblem() {

    //cout << "Beginning lanczosClass.setupProblem()." << endl;

    //A^0*q = q = v_1 = input vector.
    // tMat* vecPointer;
    // vecPointer = sysObjPtr->getStateDevPtr(this->vecInNum);
    int colNum = 1;

    //Copy state vector to column of matQ
    if( ! this->matQ.copyForeignDeviceVecToDeviceMatColumn( 
        // vecPointer,
        this->ptrVecIn,
        colNum )
    ) return false;



    //If using Taylor-Krylov method
    // if ( this->useTaylorKrylov ) {
    //     //Copy state vector to column of matNaiveKrylov
    //     if(!
    //     this->matNaiveKrylov.copyForeignDeviceVecToDeviceMatColumn( 
    //         // vecPointer, 
    //         this->ptrVecIn,
    //         colNum )
    //     ) return false;
    // }   

    return true;

}



//Run the lanczos algorithm with creating the Taylor-like subspace
template <typename tMat, typename tState, typename tReal>
bool lanczosClass<tMat,tState,tReal>::
runStandardLanczos(){

    //cout << "Beginning lanczosClass.runStandardLanczos()" << endl;

    cublasStatus_t cbStatus;

    int j;

    for(j=1; j < this->m+1; j++) {
        

        //cout << "Beginning step 4." << endl;
        //4. w = A*q_j
        if(!
        sparseMatDenseVecMult(
            *(this->d_matCsr), 
            this->matQ.getDeviceColumnPtr(j),
            thrust::raw_pointer_cast(&this->d_w[0])
            )
        ) return false;


        //cout << "Beginning step 5." << endl;
        //5. alphaj = dot(q_j,w)
        cbStatus = cublasZdotu( //result = x dot y
            this->sysObjPtr->cbHandle,                  //handle
            this->N,                                    //length of each vector
            this->matQ.getDeviceColumnPtr(j),           //x
            1,                                          //stride
            thrust::raw_pointer_cast(&this->d_w[0]),    //y
            1,                                          //stride
            thrust::raw_pointer_cast(&this->d_alphas[j])     //result
            );
        if(cbStatus != CUBLAS_STATUS_SUCCESS) { 
            cout << "ERROR in runStandardLanczos in step 5. ";
            cout << "Aborting." << endl;
            return false;
        }


        //cout << "Beginning step 6 prep." << endl;
        //Prep for 6. Make alphas negative
        negateSingle<<<1,1>>>(
            thrust::raw_pointer_cast(&this->d_negAlphas[0]),
            thrust::raw_pointer_cast(&this->d_alphas[0]),
            j
            );


        //cout << "Beginning step 6." << endl;
        //6. w = w - alpha_j*q_j - beta_j*q_(j-1) [q_0=0]
        cbStatus = cublasZaxpy( //y = y + const*x .... w = w - alpha_j*q_j
            this->sysObjPtr->cbHandle,                          //handle
            this->N,                                        //length of vector
            thrust::raw_pointer_cast(&this->d_negAlphas[j]),//multiplier (will this cause memory-access problems??)
            this->matQ.getDeviceColumnPtr(j),               //x
            1,  //stride
            thrust::raw_pointer_cast(&this->d_w[0]),        //y
            1 //stride
            );
        if(cbStatus != CUBLAS_STATUS_SUCCESS) {
            cout << "ERROR in runStandardLanczos in step 6 pt1. ";
            cout << "Aborting." << endl;
            return false;
        }
        cbStatus = cublasZaxpy( //y = y + multiplier*x .... w = w - beta_(j-1)*q_(j-1)
            this->sysObjPtr->cbHandle,                          //handle
            this->N,                                        //length of vector
            thrust::raw_pointer_cast(&this->d_negBetas[j-1]),//multiplier (will this cause memory-access problems??)
            this->matQ.getDeviceColumnPtr(j-1),               //x
            1,  //stride
            thrust::raw_pointer_cast(&this->d_w[0]),        //y
            1 //stride
            );
        if(cbStatus != CUBLAS_STATUS_SUCCESS) {
            cout << "ERROR in runStandardLanczos in step 6 pt2. ";
            cout << "Aborting." << endl;
            return false;
        }


        //cout << "Beginning step 7." << endl;
        //7. beta_j = ||w|| (L2-norm)
        cbStatus = cublasDznrm2(
            this->sysObjPtr->cbHandle,                      //Handle
            this->N,                                    //vector length
            thrust::raw_pointer_cast(&this->d_w[0]),    //vector
            1,      //stride
            (double*) thrust::raw_pointer_cast(&this->d_betas[j])
        );

        //cout << "Beginning step 8." << endl;
        //Prep for 8. Do beta_j's inverse
        invertSingle<<<1,1>>>(
            thrust::raw_pointer_cast(&this->d_invBetas[0]),
            thrust::raw_pointer_cast(&this->d_betas[0]),
            j
        );


        //8. q_(j+1) = w / beta_m
        cbStatus = cublasZcopy(
            this->sysObjPtr->cbHandle,
            this->N,                 //vector length
            thrust::raw_pointer_cast(&this->d_w[0]),
            1,  //stride
            matQ.getDeviceColumnPtr(j+1),
            1   //stride
        );
        cbStatus = cublasZscal(
            this->sysObjPtr->cbHandle,
            this->N,
            thrust::raw_pointer_cast(&this->d_invBetas[j]),
            matQ.getDeviceColumnPtr(j+1),
            1 //stride
        );


        //Prepare for next iteration
        negateSingle<<<1,1>>>(
            thrust::raw_pointer_cast(&this->d_negBetas[0]),
            thrust::raw_pointer_cast(&this->d_betas[0]),
            j
        );





    }


    return true;

}



//Find ||A*v_{m+1}||_2, used in error analysis
template <typename tMat, typename tState, typename tReal>
bool lanczosClass<tMat,tState,tReal>::
find_norm2_A_v_mp1() {


    //Put result into w
    //w = A*q_m+1
    if(!
    sparseMatDenseVecMult(
        *(this->d_matCsr), 
        this->matQ.getDeviceColumnPtr(this->m+1),
        thrust::raw_pointer_cast(&this->d_w[0])
        )
    ) return false;

    //Temporarily switch handle's pointer mode
    cublasSetPointerMode(this->sysObjPtr->cbHandle, CUBLAS_POINTER_MODE_HOST);
    
    //Get the norm
    //cublasStatus_t cbStatus;
    //cbStatus = 
    cublasDznrm2(
        this->sysObjPtr->cbHandle,                      //Handle
        this->N,                                    //vector length
        thrust::raw_pointer_cast(&this->d_w[0]),    //vector
        1,      //stride
        &(this->norm2_A_v_mp1)
    );

    //Switch pointer mode back
    cublasSetPointerMode(this->sysObjPtr->cbHandle, CUBLAS_POINTER_MODE_DEVICE);

    return true;

}







//NOT TO BE USED ANYMORE
//Create taylor-type expansion of Krylov space
// template <typename tMat, typename tState, typename tReal>
// bool lanczosClass<tMat,tState,tReal>::
// createKrylovSpace() {


//     int k;

//     //DO YOU HAVE TO WORRY ABOUT SYNCHRONIZATION AT ALL?

//     //Loop to form each A^k*q
//     for(k=2; k <= this->matNaiveKrylov.numCols; k++) {

//         if( ! this->do_Apowk_times_input(k) ){
//             return false;
//         }
//         //sparseMatDenseVecMult()

//     }

//     return true;

// }



//Go from naive Krylov space to the lanczos decomposition
template <typename tMat, typename tState, typename tReal>
void lanczosClass<tMat,tState,tReal>::
lanczosDecompFromNaiveKrylov() {




}



//Single matrix multiplication
template <typename tMat, typename tState, typename tReal>
bool lanczosClass<tMat,tState,tReal>::
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
        d_one,                              //constant
        d_csrMat.cuspMatDescr,              //matrix descriptor
        d_csrMat.csrValA,                   //values in csr matrix
        d_csrMat.csrRowPtrA,                   //csrRowPtrA
        d_csrMat.csrColIndA,                   //csrColIndA
        inVec,                              //x-vector
        d_zero,                             //the beta-value
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




//NOT TO BE USED ANYMORE
//Calculate one member of Kyrlov subspace
// template <typename tMat, typename tState, typename tReal>
// bool lanczosClass<tMat,tState,tReal>::
// do_Apowk_times_input(int k) {

//     cusparseStatus_t csStatus;
//     csStatus = cusparseZcsrmv( //y = alpha*op(A)*x + beta*y //ASYNCHRONOUS!

//         this->sysObjPtr->csHandle,     //handle
//         CUSPARSE_OPERATION_NON_TRANSPOSE,   //operation
//         this->N,                            //Rows in mat A
//         this->N,                            //Columns in mat A
//         this->d_matCsr->nnz,          //nnz
//         d_one,                              //constant
//         this->d_matCsr->cuspMatDescr, //matrix descriptor
//         this->d_matCsr->csrValA,      //values in csr matrix
//         this->sysObjPtr->d_csrRowPtrA,          //csrRowPtrA
//         this->sysObjPtr->d_csrColIndA,          //csrColIndA
//         this->matNaiveKrylov.getDeviceColumnPtr(k-1), //x-vector
//         d_zero, //the beta-value
//         this->matNaiveKrylov.getDeviceColumnPtr(k) //y-vector (result)
//     );
//     if(csStatus!=CUSPARSE_STATUS_SUCCESS) {
//         cout << "ERROR in do_Apowk_times_input for k = " << k << endl;
//         return false;
//     }

//     //Synchronize, because csrmv() is asynchronous
//     cudaDeviceSynchronize();

//     return true;

// }


//Copy back and return norm, only if that is the mode it was in
template <typename tMat, typename tState, typename tReal>
tMat lanczosClass<tMat,tState,tReal>::
copyBackAndGetNorm() {

    if( ! this->storingNormAndScaling ) {
        cout << "Warning. Specified not to store norm in lanczosClass, so should not ";
        cout << "be calling copyBackAndGetNorm. Ignoring call, returning NaN." << endl;
        tMat tMatVal;
        tMatVal.x = tMatVal.y = 1./(1.-1.);
        return tMatVal;
    }

    cudaMemcpy(&(this->h_norm), d_norm, sizeof(h_norm), cudaMemcpyDeviceToHost);
    return this->h_norm;

}





//Clean up
template <typename tMat, typename tState, typename tReal>
void lanczosClass<tMat,tState,tReal>::
cleanup() {

    
}


























