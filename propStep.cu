/*
propStep.cu
Propogates one step of lanczos-decomposed system

Nicolas Sawaya
2013
*/


bool propagateSystem(
    systemClass<typeMat,typeState,typeReal> *sysObjPtr,
    lanczosClass<typeMat,typeState,typeReal> *lanObjPtr,
    padeClass<typeMat,typeState,typeReal> *padeObjPtr,
    int destVecNum
    ) {

    // cout << "sysObj.N = " << sysObj.N;
    // cout << "sysObj.m = " << sysObj.m << endl;


    cublasStatus_t cbStatus;

    typeMat *ptrQ = lanObjPtr->matQ.getDeviceColumnPtr(1);
    typeMat *ptrExp_tT = padeObjPtr->matExp_tT.getDeviceColumnPtr(0);
    //typeMat *ptrState = sysObj.matOfStates.getDeviceColumnPtr(vecInNum+1);
    typeMat *ptrState = sysObjPtr->matOfStates.getDeviceColumnPtr(destVecNum);

    // Incidentally, only first column of is relevant.
    // ***Think about how this affects operations in padeClass.
    // w(t) = beta*V*exp(t*H)*e1 => only first column of exp(t*H)
    cbStatus = cublasZgemv( //y = alpha*op(A)*x + beta*y
        sysObjPtr->cbHandle,
        CUBLAS_OP_N,
        sysObjPtr->N, //rows in mat
        sysObjPtr->m, //cols in mat
        lanObjPtr->d_one, //2-norm of original vector
        ptrQ, //The matrix (first column is just zeros)
        sysObjPtr->N, //leading dimension of matrix (number of elements per column)
        ptrExp_tT, //vector (first column of exp(tT) matrix)
        1, //stride
        lanObjPtr->d_zero, //beta-multiplier zero.
        ptrState, //result vector
        1 //stride
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR at cublasZgemv() in propagateSystem()." << endl;
        return false;
    }

    //cout << "Finished this propagation step." << endl;





    return true;

}


















