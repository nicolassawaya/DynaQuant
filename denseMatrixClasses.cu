/*
denseMatrixClasses.cu
Classes for dense matrices.
COLUMN-MAJOR FORMAT! (column-major)

Nicolas Sawaya
June 2013
Aspuru-Guzik Group
*/



#include <denseMatrixClasses.h>







/********************/
/********************/
//denseMatrixHostOnlyClass
/********************/
/********************/

//Constructor
template <typename tMat>
denseMatrixHostOnlyClass<tMat>::
denseMatrixHostOnlyClass() {

}

//Allocating
template <typename tMat>
bool denseMatrixHostOnlyClass<tMat>::
allocateOnHost(int rows, int cols) {

    //Set constants
    this->numRows = rows;
    this->numCols = cols;
    this->totalNumElem = rows*cols;
    this->memorySize = this->totalNumElem * sizeof(tMat);

    //Allocate
    this->h_fullArray = new tMat[this->totalNumElem];

    return true;
}

//Return column pointer
template <typename tMat>
tMat* denseMatrixHostOnlyClass<tMat>::
getHostColumnPtr(int colToGet) {

    //Return pointer for beginning of row. Remember this is column-major format.
    return &( this->h_fullArray[ this->numRows * colToGet ] );

}

//Normalize Host column
template <typename tMat>
bool denseMatrixHostOnlyClass<tMat>::
normalizeHostCol(int colNum) {

    //Get norm2 of column    
    double nrm2 = this->getHostColNorm2(colNum);

    //Normalize
    scaleHostCol(colNum, 1./nrm2);

    return true;

}

//Get norm2 of a column
template <typename tMat>
double denseMatrixHostOnlyClass<tMat>::
getHostColNorm2(int colNum) {

    double nrm2 = 0.;

    for(int elem=0; elem<this->numRows; elem++) {
        nrm2 += pow(this->h_fullArray[elem + colNum*this->numRows ].x, 2);
        nrm2 += pow(this->h_fullArray[elem + colNum*this->numRows ].y, 2);
    }

    return sqrt(nrm2);

}

//Scale a column
template <typename tMat>
bool denseMatrixHostOnlyClass<tMat>::
scaleHostCol(int colNum, double scaleVal) {

    for(int elem=0;elem<this->numRows;elem++) {
        this->h_fullArray[elem + colNum*this->numRows ].x *= scaleVal;
        this->h_fullArray[elem + colNum*this->numRows ].y *= scaleVal;
    }

    return true;

}


//Set a full column to zero
template <typename tMat>
void denseMatrixHostOnlyClass<tMat>::
setColumnToZero(int column) {

    for(int i=0;i<this->numRows;i++) {
        this->h_fullArray[ i + column*this->numRows ].x = 0.;
        this->h_fullArray[ i + column*this->numRows ].y = 0.;
    }

}

//Set a single host element
template <typename tMat>
void denseMatrixHostOnlyClass<tMat>::
setHostElement(int row, int col, tMat value) {
//ADD CHECK FOR INDEX BEING WITHIN RANGE!
    //Remember, column-major format
    int index = col * this->numRows + row;
    this->h_fullArray[index] = value;

}

//Get a single host element
template <typename tMat>
tMat denseMatrixHostOnlyClass<tMat>::
getHostElement(int row, int col) {

    //Remember, column-major format
    int index = col * this->numRows + row;
    //cout << "index, totalNumElem: " << index << "," << totalNumElem << endl;
    return this->h_fullArray[index];

}


//Print host matrix
template <typename tMat>
void denseMatrixHostOnlyClass<tMat>::
printHostMat() {

    cout << this->numRows << " by " << this->numCols << " matrix." << endl;

    cout << setprecision(4) << endl;

    for( int row=0;row<this->numRows;row++ ) {
        for( int col=0;col<this->numCols;col++ ) {

            int index = col*this->numRows + row;

            cout << setw(9);
            cout << this->h_fullArray[index].x;
            cout << "+";
            cout << setw(4);
            cout << this->h_fullArray[index].y;
            cout << "*i;";

        }
        cout << endl;
    }

    cout << endl;



}









/********************/
/********************/
//denseMatrixDeviceOnlyClass
/********************/
/********************/

//Constructor
template <typename tMat>
denseMatrixDeviceOnlyClass<tMat>::denseMatrixDeviceOnlyClass() {

}

//Allocate on device
template <typename tMat>
bool denseMatrixDeviceOnlyClass<tMat>::allocateOnDevice(int rows, int cols) {

    //cout << "Beginning allocateOnDevice()." << endl;

    //Set constants
    this->numRows = rows;
    this->numCols = cols;
    this->totalNumElem = rows*cols;
    this->memorySize = this->totalNumElem * sizeof(tMat);
    this->columnMemorySize = rows * sizeof(tMat);

    //Allocate
    if( cudaMalloc((void**)&(this->d_fullArray),this->memorySize) !=cudaSuccess) {
        cout << "ERROR in allocating space in denseMatrixDeviceOnlyClass<tMat>::allocateOnDevice().";
        cout << " Aborting" << endl;
        return false;
    }

    return true;
}


//Return device pointer for a given column
template <typename tMat>
tMat* denseMatrixDeviceOnlyClass<tMat>::
getDeviceColumnPtr(int colToGet) {

    //cout << "Beginning getDeviceColumnPtr()" << endl;

    //Return pointer for beginning of row. Remember this is column-major format.

    if(colToGet >= this->numCols) {
        cout << "ERROR!!! colToGet < this->numCols in denseMatrixDeviceOnlyClass::getDeviceColumnPtr.";
        cout << " No action taken." << endl;
    }

    return &( this->d_fullArray[ this->numRows * colToGet ] );


}


//Retrun device pointer for a specific element
template <typename tMat>
tMat* denseMatrixDeviceOnlyClass<tMat>::
getDeviceElementPtr(int row, int col) {

    if( col >= this->numCols || row >= this->numRows ) {
        cout << "ERROR in getDeviceColumnPtr(). Row or col out of bounds." << endl;
        cout << "this->numCols, input col, this->numRows, input row: " << endl;
        cout << this->numCols << "," << col << "," << this->numRows << "," << row << endl;
        cout << "No action taken." << endl;
    }

    return &( this->d_fullArray[ this->numRows * col  +  row ] );

}


//copy a column from a host matrix
template <typename tMat>
bool denseMatrixDeviceOnlyClass<tMat>::
copyColumnFromHost(denseMatrixHostOnlyClass<tMat> hostMat, int column) {

    int startIndex = this->numRows * column;

    if( cudaMemcpy( &(this->d_fullArray[startIndex]), &(hostMat.h_fullArray[startIndex])
    , this->columnMemorySize, cudaMemcpyHostToDevice) !=cudaSuccess ){
        cout << "ERROR in copying from host to device in copyColumnFromHost().";
        cout << " Aborting." << endl;
        return false;
    }

    return true;

}


//Copy a column from somewhere else
template <typename tMat>
bool denseMatrixDeviceOnlyClass<tMat>::
copyForeignDeviceVecToDeviceMatColumn(tMat* vecPointer, int column) {

    cudaError_t cuErr;
    //cublasStatus_t cbStat;

    tMat* matColPtr = &(this->d_fullArray[ column*this->numRows ]);

    cuErr = cudaMemcpy( matColPtr, vecPointer, this->columnMemorySize, cudaMemcpyDeviceToDevice );
    if(cuErr!=cudaSuccess) {
        cout << "ERROR in copyForeignDeviceVecToDeviceMatColumn(). ";
        cout << "Error is: " << cudaGetErrorString(cuErr) << ". Aborting." << endl;
        return false;
    }

    //Synchronize I think is necessary
    //cudaDeviceSynchronize();


    // cbStat = cublasZcopy(
    //     this->cbHandle,
    //     this->numRows,                 //vector length
    //     vecPointer,    //source
    //     1,  //stride
    //     matColPtr,           //destination
    //     1   //stride
    // );
    // if(cbStat!=CUBLAS_STATUS_SUCCESS) {
    //     cout << "Error in copyForeignDeviceVecToDeviceMatColumn(). ";
    //     cout << "Error is: " << cublasGetErrorString(cbStat) << ". Aborting." << endl;
    //     return false;
    // }



    return true;

}


//Copy a host column from somewhere else
template <typename tMat>
bool denseMatrixDeviceOnlyClass<tMat>::
copyForeignHostVecToDeviceMatColumn(tMat* vecPointer, int column) {

    cudaError_t cuErr;

    tMat* matColPtr = &(this->d_fullArray[ column*this->numRows ]);

    cuErr = cudaMemcpy( matColPtr, vecPointer, this->columnMemorySize, cudaMemcpyHostToDevice );
    if(cuErr!=cudaSuccess) {
        cout << "ERROR in copyForeignHostVecToDeviceMatColumn(). ";
        cout << "Error is: " << cudaGetErrorString(cuErr) << ". Aborting." << endl;
        return false;
    }

    return true;

}









/********************/
/********************/
//denseMatrixClass
/********************/
/********************/


//Constructor
template <typename tMat>
denseMatrixClass<tMat>::denseMatrixClass() {

}



//Allocate on host and device
template <typename tMat>
bool denseMatrixClass<tMat>::allocateOnHostAndDevice(int rows, int cols) {

    //Allocate
    if( ! this->deviceMat.allocateOnDevice(rows,cols) ) return false;
    this->hostMat.allocateOnHost(rows,cols);


    this->numRows = this->hostMat.numRows;
    this->numCols = this->hostMat.numCols;
    this->totalNumElem = this->hostMat.totalNumElem;
    this->memorySize = this->memorySize;


    return true;

}




//Return column pointer
template <typename tMat>
tMat* denseMatrixClass<tMat>::
getDeviceColumnPtr(int colToGet) {

    return this->deviceMat.getDeviceColumnPtr(colToGet);

}

//Return element pointer
template <typename tMat>
tMat* denseMatrixClass<tMat>::
getDeviceElementPtr(int row, int col) {

    return this->deviceMat.getDeviceElementPtr(row,col);

}


//Return column pointer
template <typename tMat>
tMat* denseMatrixClass<tMat>::
getHostColumnPtr(int colToGet) {

    return this->hostMat.getHostColumnPtr(colToGet);

}


//Set a give column to all zeros
template <typename tMat>
bool denseMatrixClass<tMat>::
setColumnToZero(int column) {

    this->hostMat.setColumnToZero(column);
    this->deviceMat.copyColumnFromHost(this->hostMat, column);

    return true;

}


//Copy in a column from somewhere else. CHANGES ONLY DEVICE MAT.
template <typename tMat>
bool denseMatrixClass<tMat>::
copyForeignDeviceVecToDeviceMatColumn(tMat* vecPointer, int column) {

    return this->deviceMat.copyForeignDeviceVecToDeviceMatColumn(vecPointer, column);

}


//Copy in a column from a host position. CHANGES ONLY DEVICE MAT.
template <typename tMat>
bool denseMatrixClass<tMat>::
copyForeignHostVecToDeviceMatColumn(tMat* vecPointer, int column) {

    return this->deviceMat.copyForeignHostVecToDeviceMatColumn(vecPointer, column);

}



//Copy a host column into the same device column
template <typename tMat>
bool denseMatrixClass<tMat>::
copyHostColToDeviceCol(int column) {

    tMat* hostPtr = this->getHostColumnPtr(column);
    tMat* devPtr = this->getDeviceColumnPtr(column);
    size_t memSize = this->deviceMat.columnMemorySize;

    cudaMemcpy( devPtr, hostPtr, memSize, cudaMemcpyHostToDevice );

    return true;

}


//Copy a device column into the same host column
template <typename tMat>
bool denseMatrixClass<tMat>::
copyDeviceColToHostCol(int column) {

    tMat* hostPtr = this->getHostColumnPtr(column);
    tMat* devPtr = this->getDeviceColumnPtr(column);
    size_t memSize = this->deviceMat.columnMemorySize;

    cudaMemcpy( hostPtr, devPtr, memSize, cudaMemcpyDeviceToHost );

    return true;

}

//Copy multiple device columns into the same host columns
template <typename tMat>
bool denseMatrixClass<tMat>::
copyDeviceColsToHostColsMultiple(int colStart, int numCols) {

    tMat* hostPtr = this->getHostColumnPtr(colStart);
    tMat* devPtr = this->getDeviceColumnPtr(colStart);
    size_t memSize = (numCols) * this->deviceMat.columnMemorySize;

    cudaMemcpy( hostPtr, devPtr, memSize, cudaMemcpyDeviceToHost );

    return true;

}


//Normalize first column on host
template <typename tMat>
bool denseMatrixClass<tMat>::
normalizeHostCol(int colNum) {

    return this->hostMat.normalizeHostCol(colNum);

}



//Set host element value
template <typename tMat>
void denseMatrixClass<tMat>::setHostElement(int row, int col, tMat value) {

    this->hostMat.setHostElement(row, col, value);

}

//Get host element value
template <typename tMat>
tMat denseMatrixClass<tMat>::
getHostElement(int row, int col) {

    return this->hostMat.getHostElement(row, col);

}


//Copy device matrix to host matrix
template <typename tMat>
bool denseMatrixClass<tMat>::
copyThisDeviceToThisHost() {

    tMat* dPtr = this->deviceMat.getDeviceColumnPtr(0);
    tMat* hPtr = this->hostMat.getHostColumnPtr(0);

    cudaMemcpy( hPtr, dPtr, this->deviceMat.memorySize, cudaMemcpyDeviceToHost );

    return true;
}


//Copy host matrix to device matrix
template <typename tMat>
bool denseMatrixClass<tMat>::
copyThisHostToThisDevice() {

    tMat* dPtr = this->deviceMat.getDeviceColumnPtr(0);
    tMat* hPtr = this->hostMat.getHostColumnPtr(0);

    cudaMemcpy( dPtr, hPtr, this->deviceMat.memorySize, cudaMemcpyHostToDevice );

    return true;

}


//Print host matrix
template <typename tMat>
void denseMatrixClass<tMat>::printHostMat() {

    this->hostMat.printHostMat();


}





















