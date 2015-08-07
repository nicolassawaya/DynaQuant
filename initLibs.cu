/*
initLibs.cu
Initialize libraries to use.

Nicolas Sawaya
2013
*/



bool initCula() {
    culaStatus culaStat;
    
    culaStat = culaInitialize();
    if(culaStat!=culaNoError) {
        cout << "CULA failed to initialize. " << endl;;
        cout << "culaGetStatusString(culaStat) = " << culaGetStatusString(culaStat);
        cout << endl;
        cout << "culaGetErrorInfo() = " << culaGetErrorInfo();
        //Apparently above returns error integer equivalent to LAPACK error
        cout << ". Aborting." << endl;
        return false;
    }
    return true;
    
}



bool initCusparse(cusparseHandle_t *cusparseHandle) {
    cusparseStatus_t csStatus;
    csStatus = cusparseCreate(cusparseHandle);
    if (csStatus != CUSPARSE_STATUS_SUCCESS) {
        cout << "Error from cusparseCreate(). csStatus = ";
        cout << cusparseGetErrorString(csStatus) << endl;
        return false;
    }
    //Set pointer mode to device
    cusparseSetPointerMode(*cusparseHandle, CUSPARSE_POINTER_MODE_DEVICE);
    return true;
}


bool initCublas(cublasHandle_t *cublasHandle) {
    cublasStatus_t cbStatus;
    //cublasHandle = 0;
    cbStatus = cublasCreate(cublasHandle);
    if (cbStatus != CUBLAS_STATUS_SUCCESS) {
        cout << "Error from cublasCreate(). cbStatus = ";
        cout << cublasGetErrorString(cbStatus) << endl;
        return false;
    }
    //Set pointer mode to device
    cbStatus = cublasSetPointerMode(*cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

    return true;
}


bool initCufft(cufftHandle *cufftPlan, int arrlength) {

    //cufftPlan1d(plan, nx, cuffttype, batch)
    if( cufftPlan1d(cufftPlan, arrlength, CUFFT_Z2Z, 1) != CUFFT_SUCCESS ) {
        cout << "ERROR in initCufft with cufftPlan1d(). Aborting." << endl;
        return false;
    }

    return true;

}




















