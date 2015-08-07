/*
denseMatrixClasses.h

Nicolas Sawaya
Aspuru-Guzik Group
July 2013
*/






/********************/
/********************/
/********************/
//Dense matrix class only on Host
template <typename tMat>
class denseMatrixHostOnlyClass {

public:
    denseMatrixHostOnlyClass();
    bool allocateOnHost(int rows, int cols);
    tMat* getHostColumnPtr(int colToGet);

    //Operations
    bool normalizeHostCol(int colNum);
    double getHostColNorm2(int colNum);
    bool scaleHostCol(int colNum, double scaleVal);
    void setColumnToZero(int column);

    void setHostElement(int row, int col, tMat value);
    tMat getHostElement(int row, int col);

    void printHostMat();

    tMat* h_fullArray;

    int numRows;
    int numCols;
    int totalNumElem;
    size_t memorySize;

};


/********************/
/********************/
/********************/
//Dense Matrix class only on Device
template <typename tMat>
class denseMatrixDeviceOnlyClass {

public:
    denseMatrixDeviceOnlyClass();
    bool allocateOnDevice(int rows, int cols);
    tMat* getDeviceColumnPtr(int colToGet);
    tMat* getDeviceElementPtr(int row, int col);
    
    bool copyColumnFromHost(denseMatrixHostOnlyClass<tMat> hostMat, int column);
    bool copyForeignDeviceVecToDeviceMatColumn(tMat* vecPointer, int column);
    bool copyForeignHostVecToDeviceMatColumn(tMat* vecPointer, int column);



    tMat* d_fullArray;

    int numRows;
    int numCols;
    int totalNumElem;
    size_t memorySize;
    size_t columnMemorySize;

};



/********************/
/********************/
/********************/
//Dense Matrix class including both device and host matrices
template <typename tMat>
class denseMatrixClass {

    public:
        denseMatrixClass();
        //bool allocateOnHost(int rows, int cols);
        //bool allocateOnDevice(int rows, int cols);
        bool allocateOnHostAndDevice(int rows, int cols);
        tMat* getDeviceColumnPtr(int colToGet);
        tMat* getDeviceElementPtr(int row, int col);
        tMat* getHostColumnPtr(int colToGet);
        bool copyHostColToDeviceCol(int column);
        bool copyDeviceColToHostCol(int column);
        bool copyDeviceColsToHostColsMultiple(int colStart, int numCols);

        //Operations
        bool normalizeHostCol(int colNum);



        bool setColumnToZero(int col);
        bool copyForeignDeviceVecToDeviceMatColumn(tMat* vecPointer, int column);
        bool copyForeignHostVecToDeviceMatColumn(tMat* vecPointer, int column);

        void setHostElement(int row, int col, tMat value);
        tMat getHostElement(int row, int col);
        

        bool copyThisDeviceToThisHost();
        bool copyThisHostToThisDevice();
        void printHostMat();


        denseMatrixDeviceOnlyClass<tMat> deviceMat;
        denseMatrixHostOnlyClass<tMat> hostMat;

        int numRows;
        int numCols;
        int totalNumElem;
        size_t memorySize;


        bool hostAllocated;
        bool deviceAllocated;



};





















