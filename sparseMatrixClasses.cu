/*
sparseMatrixClasses.cu
Defines objects for coo and csr formats.

Nicolas Sawaya
2013
*/


template <typename tMat>
class matCooClass {
    
    public:
        matCooClass();
        // void setCusparseHandle(cusparseHandle_t &csHandle);
        // void setCublasHandle(cublasHandle_t &cbHandle);
        void setCusparseHandle(cusparseHandle_t *csHandle);
        void setCublasHandle(cublasHandle_t *cbHandle);

        bool readInFile(string filename, bool requireAllDiagElems);
        bool createOnDevice(matCooClass<tMat> deviceMatrix);
        bool copyFromDevice(matCooClass<tMat> deviceMatrix); //For use with a host object
        bool areAllDiagonalsPresent();
        void freeMemory();


        int nnz;
        int N; //This is an NxN matrix
        tMat* cooValA;
        int* cooRowIndA;
        int* cooColIndA;

        size_t valSize;
        size_t indSize;
        
        cusparseMatDescr_t cuspMatDescr;
        /*
        MatrixType
        FillMode
        DiagType
        IndexBase
        */

        bool isOnDevice;

        cusparseHandle_t csHandle;

        
};




//Constructor
template <typename tMat>
matCooClass<tMat>::matCooClass() {
    if( cusparseCreateMatDescr(&this->cuspMatDescr)!=CUSPARSE_STATUS_SUCCESS ){
        cout << "ERROR using cusparseCreateMatDescr() in matCooClass constructor!" << endl;
    }
}


//Set cusparse handle
template <typename tMat>
void matCooClass<tMat>::setCusparseHandle(cusparseHandle_t *csHandle){

    this->csHandle = *csHandle;

}



//Reading in the file
template <typename tMat>
bool matCooClass<tMat>::readInFile(string filename, bool requireAllDiagElems=false){

    //Matrix resides on host
    this->isOnDevice = false;


    //Open file
    ifstream inFile;
    inFile.open( filename.c_str() );
    if (! inFile.is_open() ) {
        cout << "ERROR. Matrix file '" << filename << "' failed to open." << endl;
        return false;
    }

    string line;
    stringstream ss;

    //Can add stuff for comments later on

    //Read matrix type
    getline(inFile, line);
    string strDiscard, strValueType, strMatrixType;
    ss.str(line);
    ss >> strDiscard; //"%%MatrixMarker"
    ss >> strDiscard; //"matrix"
    ss >> strDiscard; //"coordinate"
    ss >> strValueType; //real, complex, integer, pattern
    ss >> strMatrixType; //general, symmetric, skew-symmetrix, Hermitian




    //Assign matrix properties
    //MAKE THESE APPROPRIATE FOR MORE GENERAL INPUTS
    cusparseSetMatFillMode(this->cuspMatDescr, CUSPARSE_FILL_MODE_LOWER); //Default for now, but be careful
    cusparseSetMatDiagType(this->cuspMatDescr, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(this->cuspMatDescr, CUSPARSE_INDEX_BASE_ZERO);
    if(strMatrixType=="general"){
        cusparseSetMatType(this->cuspMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cout << "Matrix is 'general'" << endl;
        //this->cuspMatDescr.MatrixType = CUSPARSE_MATRIX_TYPE_GENERAL;
    } else if (strMatrixType=="symmetric") {
        cusparseSetMatType(this->cuspMatDescr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
        cout << "Matrix is 'symmetric'" << endl;
        //this->cuspMatDescr.MatrixType = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
    } else if (strMatrixType=="hermitian") {
        cusparseSetMatType(this->cuspMatDescr, CUSPARSE_MATRIX_TYPE_HERMITIAN);
        cout << "Matrix is 'hermitian'" << endl;
        //this->cuspMatDescr.MatrixType = CUSPARSE_MATRIX_TYPE_HERMITIAN;
    } else {
        cout << "ERROR reading matrix file. Unknown matrix type. Aborting." << endl;
        return false;
    }


    //Reading whether base zero or base one
    bool fileIsBaseOne;
    getline(inFile, line);
    if (line=="BASE_ZERO") {
        fileIsBaseOne = false;
    } else if (line=="BASE_ONE") {
        fileIsBaseOne = true;
    } else {
        cout << "ERROR. Must specify 'BASE_ZERO' or 'BASE_ONE' on second line ";
        cout << "of matrix file. Aborting." << endl;
        return false;
    }



    //Reading N and nnz
    stringstream ss1;
    getline(inFile, line);
    ss1.str(line);
    ss1 >> this->N;
    ss1 >> strDiscard; //ASSUMING SQUARE MATRIX FOR NOW
    ss1 >> this->nnz;

    // cout << "N = " << this->N << endl;
    // cout << "nnz = " << this->nnz << endl;

    if( this->N <= 0 || this->nnz <= 0) {
        cout << "ERROR in readInFile(). this->N = " << this->N;
        cout << ", and this->nnz = " << this->nnz << "." << endl;
        return false;
    }

    //Set memory sizes
    this->valSize = this->nnz * sizeof(typeof(*(this->cooValA)));
    this->indSize = this->nnz * sizeof(int);


    //Allocate for arrays
    this->cooValA = new tMat[this->nnz];
    this->cooRowIndA = new int[this->nnz];
    this->cooColIndA = new int[this->nnz];

    //If needed, prepare to check for diagonals
    bool* diagPresent;
    if(requireAllDiagElems) {
        diagPresent = new bool[this->N];
        for(int i=0;i<this->N;i++) {
            diagPresent[i] = false;
        }
    }


    //Read matrix contents
    int counter = 0;
    getline(inFile,line);
    while (inFile.good()) {

        stringstream ss;
        ss.str(line);
        if(counter >= this->nnz) {
            cout << "WARNING. counter >= nnz. No error thrown." << endl;
            cout << "Discounted line:" << endl;
            cout << line << endl << endl;
        } else {
            ss >> cooRowIndA[counter];
            ss >> cooColIndA[counter];
            if(fileIsBaseOne){
                cooRowIndA[counter] --;
                cooColIndA[counter] --;
            }
            if(requireAllDiagElems){
                if(cooRowIndA[counter]==cooColIndA[counter]) {
                    diagPresent[cooRowIndA[counter]] = true;
                }
            }
            ss >> cooValA[counter].x; //x is the real portion
            cooValA[counter].y = 0; //For now all Hamiltonians of interest are real
            //cout << cooValA[counter].x << endl;
        }

        getline(inFile,line);
        counter++;

        // if(counter < this->nnz) {
        //     ss >> cooRowIndA[counter];
        //     ss >> cooColIndA[counter];
        //     if(fileIsBaseOne){
        //         cooRowIndA[counter] --;
        //         cooColIndA[counter] --;
        //     }
        //     if(requireAllDiagElems){
        //         if(cooRowIndA[counter]==cooColIndA[counter]) {
        //             diagPresent[cooRowIndA[counter]] = true;
        //         }
        //     }
        //     ss >> cooValA[counter].x; //x is the real portion
        //     cooValA[counter].y = 0; //For now all Hamiltonians of interest are real
        //     //cout << cooValA[counter].x << endl;
        // } else {
        //     cout << "WARNING. counter > nnz. No error thrown." << endl;
        //     cout << "Discounted line:" << endl;
        //     cout << line << endl << endl;
        // }
        // counter++;
    }


    //Check for diagonals, if any are missing then throw error
    if(requireAllDiagElems) {
        for(int elem=0;elem<this->N;elem++) {
            if(diagPresent[elem]==false){
                cout << "ERROR. All diagonals (including zeros) are required to be present. ";
                cout << "Diagonal " << elem << " (base zero) is missing. Add a zero in its place ";
                cout << "in the correct position in the file, ";
                cout << "and remember to update nnz at the top of the file." << endl;
                cout << "Aborting." << endl;
                delete[] diagPresent;
                return false;
            }
        }
        delete[] diagPresent;
    }


    if(counter < this->nnz-1) {
        cout << "ERROR at matCooClass::readInFile. ";
        cout << "At loop's completion, counter = " << counter;
        cout << ", but this->nnz = " << this->nnz << ". Aborting." << endl;
        return false;
    }




    return true;


}


//Check to see that all diagonals are stored
template <typename tMat>
bool matCooClass<tMat>::
areAllDiagonalsPresent() { 

    //Very naive procedure

    //bool diagPresent[this->N];

    // int diagToCheck=0;
    // for(int elem=0; elem < this->nnz; elem++) {



    //     //" (zero-based)"

    //     if(elem==this->nnz) break;

    // }

    // for(int diag=0;diag<this->nnz;diag++) {

    // }



    return true;


}



//This object becomes a device matrix, copied from a different host matrix
template <typename tMat>
bool matCooClass<tMat>::
createOnDevice(matCooClass<tMat> hostMatrix) {

    //Matrix resides on gpu
    this->isOnDevice = true;

    //cudaError_t cudaErr;

    //Copy constants between the two objects
    this->nnz = hostMatrix.nnz;
    this->N = hostMatrix.N;
    this->valSize = hostMatrix.valSize;
    this->indSize = hostMatrix.indSize;
    cusparseSetMatFillMode(this->cuspMatDescr, cusparseGetMatFillMode(hostMatrix.cuspMatDescr) ); //Default for now, but be careful
    cusparseSetMatDiagType(this->cuspMatDescr, cusparseGetMatDiagType(hostMatrix.cuspMatDescr) );
    cusparseSetMatIndexBase(this->cuspMatDescr, cusparseGetMatIndexBase(hostMatrix.cuspMatDescr) );
    cusparseSetMatType(this->cuspMatDescr, cusparseGetMatType(hostMatrix.cuspMatDescr) );

    //Allocate memory on device
    //int memsizeVal = this->nnz * sizeof(typeof(*(this->cooValA)));
    if( cudaMalloc( (void**) &(this->cooValA) , this->valSize ) !=cudaSuccess){
        cout << "ERROR in cudaMalloc in 'createOnDevice()' for cooValA.";
        cout << " Aborting" << endl;
        return false;
    }
    //int memsizeRow = this->nnz * sizeof(typeof(*(this->cooRowIndA)));
    if( cudaMalloc( (void**) &(this->cooRowIndA) , this->indSize ) !=cudaSuccess){
        cout << "ERROR in cudaMalloc in 'createOnDevice()' for cooRowIndA.";
        cout << " Aborting" << endl;
        return false;
    }
    //int memsizeCol = this->nnz * sizeof(typeof(*(this->cooColIndA)));
    if( cudaMalloc( (void**) &(this->cooColIndA) , this->indSize ) !=cudaSuccess){
        cout << "ERROR in cudaMalloc in 'createOnDevice()' for cooColIndA.";
        cout << " Aborting" << endl;
        return false;
    }


    //Copy over
    if( cudaMemcpy( this->cooValA, hostMatrix.cooValA, this->valSize, cudaMemcpyHostToDevice) !=cudaSuccess ) {
        cout << "ERROR copying cooValA in 'createOnDevice().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->cooRowIndA, hostMatrix.cooRowIndA, this->indSize, cudaMemcpyHostToDevice) !=cudaSuccess ) {
        cout << "ERROR copying cooRowIndA in 'createOnDevice().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->cooColIndA, hostMatrix.cooColIndA, this->indSize, cudaMemcpyHostToDevice) !=cudaSuccess ) {
        cout << "ERROR copying cooColIndA in 'createOnDevice().'";
        cout << " Aborting." << endl;
        return false;
    } 



    return true;


}


//Copy the elements (no indices) of the device matrix to this host matrix
//ONLY works if this instance is a host matrix
template <typename tMat>
bool matCooClass<tMat>::
copyFromDevice(matCooClass<tMat> deviceMatrix) {


    if( cudaMemcpy(this->cooValA, deviceMatrix.cooValA, this->valSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR in matCooClass::copyFromDevice" << endl;
        // cout << " Aborting." << endl;
        return false;
    }


    return true;

}





//Free memory
template <typename tMat>
void matCooClass<tMat>::freeMemory() {

    //Free memory
    if(this->isOnDevice==false) {
        delete this->cooValA;
        delete this->cooRowIndA;
        delete this->cooColIndA;
    } else {
        cudaFree(this->cooValA);
        cudaFree(this->cooRowIndA);
        cudaFree(this->cooColIndA);
    }


}



























/**********************/
//Compressed Sparse Row Class
template <typename tMat>
class matCsrClass {

    public:
        matCsrClass();
        void setCusparseHandle(cusparseHandle_t *csHandle);
        void setCublasHandle(cublasHandle_t *cbHandle);

        bool pointToCooAndConvert(matCooClass<tMat> cooMat); //both must  be on device
        bool createFromDeviceCsrMat(matCsrClass<tMat> d_csrMat);
        bool copyFromDeviceCsrMat(matCsrClass<tMat> d_csrMat);

        void printMatToScreen();
        //bool readInFile(string filename);
        void freeMemory();

        int nnz;
        int N;
        tMat* csrValA;
        int* csrRowPtrA;
        int* csrColIndA;
        cusparseMatDescr_t cuspMatDescr;

        size_t valSize;
        size_t rowPtrSize;
        size_t colIndSize;

        bool isOnDevice;

        cusparseHandle_t csHandle;

};




//Constructor
template <typename tMat>
matCsrClass<tMat>::matCsrClass() {
    if( cusparseCreateMatDescr( &this->cuspMatDescr )!=CUSPARSE_STATUS_SUCCESS ) {
        cout << "ERROR using cusparseCreateMatDescr() in matCsrClass constructor!" << endl;
    }
}

//Set cusparse handle
template <typename tMat>
void matCsrClass<tMat>::setCusparseHandle(cusparseHandle_t *csHandle){

    this->csHandle = *csHandle;

}

template <typename tMat>
bool matCsrClass<tMat>::pointToCooAndConvert(matCooClass<tMat> cooMat) {

    //Matrix resides on gpu
    this->isOnDevice = true;

    //cudaError_t cudaErr;

    //Copy constants between the two objects
    this->nnz = cooMat.nnz;
    this->N = cooMat.N;
    this->valSize = cooMat.valSize;
    this->colIndSize = cooMat.indSize;
    this->rowPtrSize = (N+1) * sizeof(int); //See cusparse manual
    cusparseSetMatFillMode(this->cuspMatDescr, cusparseGetMatFillMode(cooMat.cuspMatDescr) ); //Default for now, but be careful
    cusparseSetMatDiagType(this->cuspMatDescr, cusparseGetMatDiagType(cooMat.cuspMatDescr) );
    cusparseSetMatIndexBase(this->cuspMatDescr, cusparseGetMatIndexBase(cooMat.cuspMatDescr) );
    cusparseSetMatType(this->cuspMatDescr, cusparseGetMatType(cooMat.cuspMatDescr) );

    //Point to values and column indices (identical to those in coo format)
    this->csrValA = cooMat.cooValA;
    this->csrColIndA = cooMat.cooColIndA;

    //Allocate memory on device
    if( cudaMalloc( (void**) &(this->csrRowPtrA) , this->rowPtrSize ) !=cudaSuccess){
        cout << "ERROR in cudaMalloc in 'createOnDevice()' for csrRowPtrA.";
        cout << " Aborting" << endl;
        return false;
    }


    cusparseStatus_t csStatus;
    //Convert
    csStatus = cusparseXcoo2csr(
        this->csHandle,
        cooMat.cooRowIndA,
        this->nnz,
        this->N,
        this->csrRowPtrA,
        CUSPARSE_INDEX_BASE_ZERO
        );
    if(csStatus!=CUSPARSE_STATUS_SUCCESS) {
        cout << "ERROR in cusparseXcoo2csr() in pointToCooAndConvert()." << endl;
        cout << "error = " << cusparseGetErrorString(csStatus);
        cout << ". Aborting." << endl;
        return false;
    }


    return true;

}


//Create a host matrix from a device matrix
template <typename tMat>
bool matCsrClass<tMat>::
createFromDeviceCsrMat(matCsrClass<tMat> d_csrMat) {

    //Array is on host
    this->isOnDevice = false;

    this->nnz = d_csrMat.nnz;
    this->N = d_csrMat.N;
    this->valSize = d_csrMat.valSize;
    this->colIndSize = d_csrMat.colIndSize;
    this->rowPtrSize = d_csrMat.rowPtrSize;
    cusparseSetMatFillMode(this->cuspMatDescr, cusparseGetMatFillMode(d_csrMat.cuspMatDescr) ); //Default for now, but be careful
    cusparseSetMatDiagType(this->cuspMatDescr, cusparseGetMatDiagType(d_csrMat.cuspMatDescr) );
    cusparseSetMatIndexBase(this->cuspMatDescr, cusparseGetMatIndexBase(d_csrMat.cuspMatDescr) );
    cusparseSetMatType(this->cuspMatDescr, cusparseGetMatType(d_csrMat.cuspMatDescr) );


    //Allocate memory on host
    this->csrValA = new tMat[this->nnz];
    this->csrRowPtrA = new int[ this->rowPtrSize / sizeof(int) ];
    this->csrColIndA = new int[this->nnz];

    //Copy back to host
    if( cudaMemcpy( this->csrValA, d_csrMat.csrValA, this->valSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrValA in 'createFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->csrRowPtrA, d_csrMat.csrRowPtrA, this->rowPtrSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrRowPtrA in 'createFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->csrColIndA, d_csrMat.csrColIndA, this->colIndSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrColIndA in 'createFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 

    return true;

}



//Copy a device matrix over to this host matrix
template <typename tMat>
bool matCsrClass<tMat>::
copyFromDeviceCsrMat(matCsrClass<tMat> d_csrMat) {

    //First check to see if sizes and matrix types are compatible
    if(
        this->nnz != d_csrMat.nnz ||
        this->N != d_csrMat.N ||
        this->valSize != d_csrMat.valSize ||
        this->rowPtrSize != d_csrMat.rowPtrSize ||
        this->colIndSize != d_csrMat.colIndSize
    ) {
        cout << "ERROR in matCsrClass::copyFromDeviceCsrMat(). Dimensions don't agree." << endl;
        cout << "this->{nnz,N,valSize,rowPtrSize,colIndSize}:" << endl;
        cout << this->nnz << " " << this->N << " " << this->valSize << " ";
        cout << this->rowPtrSize << " " << this->colIndSize << endl;
        cout << "d_csrMat.{nnz,N,valSize,rowPtrSize,colIndSize}:" << endl;
        cout << d_csrMat.nnz << " " << d_csrMat.N << " " << d_csrMat.valSize << " ";
        cout << d_csrMat.rowPtrSize << " " << d_csrMat.colIndSize << endl << endl << endl;
        return false;
    }

    //Copy back to host
    if( cudaMemcpy( this->csrValA, d_csrMat.csrValA, this->valSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrValA in 'copyFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->csrRowPtrA, d_csrMat.csrRowPtrA, this->rowPtrSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrRowPtrA in 'copyFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 
    if( cudaMemcpy( this->csrColIndA, d_csrMat.csrColIndA, this->colIndSize, cudaMemcpyDeviceToHost) !=cudaSuccess ) {
        cout << "ERROR copying csrColIndA in 'copyFromDeviceCsrMat().'";
        cout << " Aborting." << endl;
        return false;
    } 



    return true;

}



//Print to screen
template <typename tMat>
void matCsrClass<tMat>::
printMatToScreen(){

    //Can print only if on host
    if(this->isOnDevice) {
        cout << "WARNING: Cannot print a matrix that resides on device. ";
        cout << "Command ignored." << endl;
        return;
    }

    cout << "CSR matrix." << endl;
    cout << this->N << "  " << this->N << "  " << this->nnz << endl;
    cout << "csrValA" << endl;
    for(int i=0; i<this->nnz; i++) {
        cout << this->csrValA[i].x << " + ";
        cout << this->csrValA[i].y << "i" << endl;
    }
    cout << "csrRowPtrA" << endl;
    for(int i=0; i<this->N+1; i++) {
        cout << this->csrRowPtrA[i] << endl;
    }
    cout << "csrColIndA" << endl;
    for(int i=0; i<this->nnz; i++) {
        cout << this->csrColIndA[i] << endl;
    }
    cout << endl;

}



//Free memory
template <typename tMat>
void matCsrClass<tMat>::
freeMemory() {

    //Free memory
    if(this->isOnDevice==false) {
        delete this->csrValA;
        delete this->csrRowPtrA;
        delete this->csrColIndA;
    } else {
        cudaFree(this->csrValA);
        cudaFree(this->csrRowPtrA);
        cudaFree(this->csrColIndA);
    }


}
















