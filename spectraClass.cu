/*
spectraClass.cu

Using scheme in Roden, Strunz, Eisfeld, JCP 134, 034902 (2011)

Nicolas sawaya
July 2013
*/






template <typename tMat, typename tState, typename tReal>
class spectraClass {
    
    public:
        spectraClass();
        spectraClass( systemClass<tMat,tState,tReal> *inputSysObjPtr );
        // bool readInDipoleFile(string &dipolesFilename);
        bool prepForAbsorb(string &dipolesFilename);
        bool prepForCD(string &dipolesFilename, string &posFilename);
        bool prepInitState(int dir);

        //Find M(t) for one section of the propagation
        bool calcCorrFunc(int stepsToDo);
        //Take FFT and copy back to host:
        bool singleSpectraCalc(int dir);
        //Average the spectra together and take transformation constant into account:
        bool completeSpectraCalc();

        //Point to system object
        systemClass<tMat,tState,tReal> *sysObjPtr;

        //To keep track of how many steps have been propagated through:
        int timestepCounter;
        

        tMat *transitionDipoles[3];
        tReal *positions[3];
        tMat *RD[3]; //RD_n = R_n cross mu_n
        tReal mu_tot[3];
        tReal RD_tot[3]; //RD_tot = sqrt( sum(RD_N) )
        thrust::device_vector<tMat> d_braForCD; // the bra (<phi0|) used for calculating time correlation in CD
        thrust::host_vector<tMat> h_braForCD;
        //tMat *absorbInitStates[3];
        // bool initStateForAbsorption(int axis);
        thrust::host_vector<tMat> h_timeCorrFunc[3]; //M(t), one for each of x,y,z
        thrust::host_vector<tMat> h_transformedFunc[3]; //FFT of M(t) to get spectra
        thrust::device_vector<tMat> d_timeCorrFunc; //Only store one on the GPU at a time
        thrust::device_vector<tMat> d_transformedFunc; //Only store one on the GPU at a time
        
        //Parameters specified in infile
        bool lineShapeDressing;
        tReal gaussLineshapeWidth; //Sigma in femtoseconds

        //Size of system
        int N;
        int totalSteps;
        int transformLength; // = totalSteps + 1

        //Used for scaling transform result
        tReal invFsToInvCm;
        tReal h_scaleTransform;
        //tReal *d_scaleTransform; //No! scaleTransform just gives the frequecy (x-axis) values! Vertical values are unchanged.

        //Absorption file names
        string absCorrFuncEachDir[3];
        string absCorrFunc;
        string absFilenamesEachDir[3];
        string absFilename;

        //CD file names
        string CDCorrFuncEachDir[3];
        string CDCorrFunc;
        string CDFilenamesEachDir[3];
        string CDFilename;


};




//Default Constructor
template <typename tMat, typename tState, typename tReal>
spectraClass<tMat,tState,tReal>::
spectraClass() {

}


//Constructor
template <typename tMat, typename tState, typename tReal>
spectraClass<tMat,tState,tReal>::
spectraClass( systemClass<tMat,tState,tReal> *inputSysObjPtr ) {
    
    //cudaError_t cuErr;




    //Point to system object
    this->sysObjPtr = inputSysObjPtr;
    this->N = inputSysObjPtr->N;
    this->totalSteps = inputSysObjPtr->totalSteps;
    this->transformLength = this->totalSteps + 1;
    this->timestepCounter = 1; //done later on now, in state prep
    this->lineShapeDressing = false;


    //file names
    //abs
    this->absCorrFuncEachDir[0] = this->sysObjPtr->outFolderName + "/absCorrFuncDir0.dat";
    cout << "this->absCorrFuncEachDir[0]" << this->absCorrFuncEachDir[0] << endl;
    this->absCorrFuncEachDir[1] = this->sysObjPtr->outFolderName + "/absCorrFuncDir1.dat";
    this->absCorrFuncEachDir[2] = this->sysObjPtr->outFolderName + "/absCorrFuncDir2.dat";
    this->absCorrFunc = this->sysObjPtr->outFolderName + "/absCorrFunc.dat";
    this->absFilenamesEachDir[0] = this->sysObjPtr->outFolderName + "/absDir0.dat";
    this->absFilenamesEachDir[1] = this->sysObjPtr->outFolderName + "/absDir1.dat";
    this->absFilenamesEachDir[2] = this->sysObjPtr->outFolderName + "/absDir2.dat";
    this->absFilename = this->sysObjPtr->outFolderName + "/absorption.dat";
    //CD
    this->CDCorrFuncEachDir[0] = this->sysObjPtr->outFolderName + "/CDCorrFuncDir0.dat";
    cout << "this->CDCorrFuncEachDir[0]" << this->CDCorrFuncEachDir[0] << endl;
    this->CDCorrFuncEachDir[1] = this->sysObjPtr->outFolderName + "/CDCorrFuncDir1.dat";
    this->CDCorrFuncEachDir[2] = this->sysObjPtr->outFolderName + "/CDCorrFuncDir2.dat";
    this->CDCorrFunc = this->sysObjPtr->outFolderName + "/CDCorrFunc.dat";
    this->CDFilenamesEachDir[0] = this->sysObjPtr->outFolderName + "/CDDir0.dat";
    this->CDFilenamesEachDir[1] = this->sysObjPtr->outFolderName + "/CDDir1.dat";
    this->CDFilenamesEachDir[2] = this->sysObjPtr->outFolderName + "/CDDir2.dat";
    this->CDFilename = this->sysObjPtr->outFolderName + "/CD.dat";

    //Allocate for transform-scaling constant
    //tReal h_scaleTransform;
    double pi = atan(1)*4;
    //should not have 2pi there:
    // this->h_scaleTransform = 2*pi / ( this->sysObjPtr->dt * this->totalSteps );
    cout << "spectraClass->h_scaleTransform: " << this->h_scaleTransform << endl;
    // cudaMalloc( (void**)&d_scaleTransform, sizeof(tReal) );
    // cudaMemcpy( &d_scaleTransform[0], &h_scaleTransform, sizeof(tReal), cudaMemcpyHostToDevice );
    // cuErr = cudaGetLastError();
    // if(cuErr!=cudaSuccess) {
    //     cout << "ERROR after cudamemcpy and cudamalloc in spectraClass constructor. ";
    //     cout << "Exiting." << endl;
    //     exit(1);
    // }

    //
    double lightspeed_meters_per_sec = 299792458;
    double meters2cm = 100;
    double sec2fs = 1.e-15;
    this->invFsToInvCm = 1. / (lightspeed_meters_per_sec * meters2cm * sec2fs);
    cout << "spectraClass->invFsToInvCm: " << this->invFsToInvCm << endl;


}



// //Read in dipole file
// template <typename tMat, typename tState, typename tReal>
// bool spectraClass<tMat,tState,tReal>::
// readInDipoleFile(string &dipolesFilename) {


// }



//Prepare for absorption calculations
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
prepForAbsorb(string &dipolesFilename) {

    //Allocate for dipoles and initital states
    this->transitionDipoles[0] = new tMat[this->N];
    this->transitionDipoles[1] = new tMat[this->N];
    this->transitionDipoles[2] = new tMat[this->N];

    //Resize vectors for M(t) and for spectra
    for(int dir=0;dir<3;dir++) {
        this->h_timeCorrFunc[dir].resize(this->transformLength);
        this->h_transformedFunc[dir].resize(this->transformLength);
    }
    this->d_timeCorrFunc.resize(this->transformLength);
    this->d_transformedFunc.resize(this->transformLength);
    tMat realOne; realOne.x = 1.; realOne.y = 0.;
    this->d_timeCorrFunc[0] = realOne;


    //Set mu_tot to zero
    mu_tot[0] = mu_tot[1] = mu_tot[2] = 0;

    //Open filestream
    ifstream inFile;
    inFile.open( dipolesFilename.c_str() );
    if ( ! inFile.is_open() ) {
        cout << "ERROR. Transition dipole parameter file failed to open. ";
        cout << "file: " << dipolesFilename << ". Aborting." << endl;
        return false;
    }


    //Variables for reading in transition dipole file
    string line; //, strParam, strVal;

    int counter=0;
    getline(inFile,line);
    while(inFile.good()) {

        if(counter >= this->N) {
            cout << "WARNING in systemClass::prepForAbsorb. counter = " << counter;
            cout << ", this-> N = ";
            cout << this->N << ". Ignoring and proceeding." << endl;
        } else {
            stringstream ss;
            ss << line;
            ss >> this->transitionDipoles[0][counter].x;
            ss >> this->transitionDipoles[1][counter].x;
            ss >> this->transitionDipoles[2][counter].x;
            this->transitionDipoles[0][counter].y = 0;
            this->transitionDipoles[1][counter].y = 0;
            this->transitionDipoles[2][counter].y = 0;

            this->mu_tot[0] += pow(transitionDipoles[0][counter].x,2)
                                + pow(transitionDipoles[0][counter].y,2);
            this->mu_tot[1] += pow(transitionDipoles[1][counter].x,2);
                                + pow(transitionDipoles[1][counter].y,2);
            this->mu_tot[2] += pow(transitionDipoles[2][counter].x,2);
                                + pow(transitionDipoles[2][counter].y,2);
            //this->h_hsr_stdDev[counter] = atof(line.c_str());
        }

        getline(inFile,line);
        counter++;

    }
    if(counter < this->N-1){
        cout << "ERROR. counter = " << counter << ", this->N = " << this->N;
        cout << ", in prepForAbsorb(). Aborting." << endl;
        return false;
    }

    //Finish mu_tot calculations
    mu_tot[0] = sqrt(mu_tot[0]);
    mu_tot[1] = sqrt(mu_tot[1]);
    mu_tot[2] = sqrt(mu_tot[2]);
    cout << endl << "mu_tot[0], mu_tot[1], mu_tot[2]:" << endl;
    cout << mu_tot[0] << "  " << mu_tot[1] << "  " << mu_tot[2] << endl;


    //Close file
    inFile.close();


    return true;



}



//Prepare for CD calculations
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
prepForCD(string &dipolesFilename, string &posFilename) {


    //Allocate for bra used in time correlation calculation
    this->d_braForCD.resize(this->N);
    this->h_braForCD.resize(this->N);

    //Allocate for dipoles and initital states
    this->transitionDipoles[0] = new tMat[this->N];
    this->transitionDipoles[1] = new tMat[this->N];
    this->transitionDipoles[2] = new tMat[this->N];

    //Allocate for positions
    this->positions[0] = new tReal[this->N];
    this->positions[1] = new tReal[this->N];
    this->positions[2] = new tReal[this->N];

    //Allocate for RD
    this->RD[0] = new tMat[this->N];
    this->RD[1] = new tMat[this->N];
    this->RD[2] = new tMat[this->N];

    //Resize vectors for M(t) and for spectra
    for(int dir=0;dir<3;dir++) {
        this->h_timeCorrFunc[dir].resize(this->transformLength);
        this->h_transformedFunc[dir].resize(this->transformLength);
    }
    this->d_timeCorrFunc.resize(this->transformLength);
    this->d_transformedFunc.resize(this->transformLength);
    tMat realOne; realOne.x = 1.; realOne.y = 0.;
    this->d_timeCorrFunc[0] = realOne;


    //Set mu_tot and RD_tot to zero
    mu_tot[0] = mu_tot[1] = mu_tot[2] = 0;
    RD_tot[0] = RD_tot[1] = RD_tot[2] = 0;

    //Open filestreams
    ifstream dipolesFile, positionsFile;
    dipolesFile.open( dipolesFilename.c_str() );
    if ( ! dipolesFile.is_open() ) {
        cout << "ERROR. Transition dipole parameter file failed to open. ";
        cout << "file: " << dipolesFilename << ". Aborting." << endl;
        return false;
    }
    positionsFile.open( posFilename.c_str() );
    if ( !positionsFile.is_open() ) {
        cout << "ERROR. Positions file failed to open. ";
        cout << "file: " << posFilename << ". Aborting." << endl;
        return false;
    }


    //Variables for reading in file
    string lineDip, linePos; //, strParam, strVal;

    int counter=0;
    getline(dipolesFile,lineDip);
    getline(positionsFile,linePos);
    //tReal crossProductResult[3];
    while(dipolesFile.good()) {

        if(counter >= this->N) {
            cout << "WARNING in systemClass::prepForAbsorb. counter = " << counter;
            cout << ", this-> N = ";
            cout << this->N << ". Ignoring and proceeding." << endl;
        } else {

            //Transition dipoles
            stringstream ss;
            ss << lineDip;
            ss >> this->transitionDipoles[0][counter].x;
            ss >> this->transitionDipoles[1][counter].x;
            ss >> this->transitionDipoles[2][counter].x;
            this->transitionDipoles[0][counter].y = 0;
            this->transitionDipoles[1][counter].y = 0;
            this->transitionDipoles[2][counter].y = 0;

            this->mu_tot[0] += pow(transitionDipoles[0][counter].x,2)
                                + pow(transitionDipoles[0][counter].y,2);
            this->mu_tot[1] += pow(transitionDipoles[1][counter].x,2);
                                + pow(transitionDipoles[1][counter].y,2);
            this->mu_tot[2] += pow(transitionDipoles[2][counter].x,2);
                                + pow(transitionDipoles[2][counter].y,2);
            //this->h_hsr_stdDev[counter] = atof(lineDip.c_str());

            //Positions
            stringstream ss1;
            ss1 << linePos;
            ss1 >> this->positions[0][counter];
            ss1 >> this->positions[1][counter];
            ss1 >> this->positions[2][counter];

            //R cross mu terms
            this->RD[0][counter].x = positions[1][counter] * 
                        (transitionDipoles[2][counter].x + transitionDipoles[2][counter].y)
                        - positions[2][counter] * 
                        (transitionDipoles[1][counter].x + transitionDipoles[1][counter].y);
            this->RD[1][counter].x = positions[2][counter] * 
                        (transitionDipoles[0][counter].x + transitionDipoles[0][counter].y)
                        - positions[0][counter] * 
                        (transitionDipoles[2][counter].x + transitionDipoles[2][counter].y);
            this->RD[2][counter].x = positions[0][counter] * 
                        (transitionDipoles[1][counter].x + transitionDipoles[1][counter].y)
                        - positions[1][counter] * 
                        (transitionDipoles[0][counter].x + transitionDipoles[0][counter].y);
            this->RD[0][counter].y = 0;
            this->RD[1][counter].y = 0;
            this->RD[2][counter].y = 0;

            //RD_tot summation
            RD_tot[0] += pow(this->RD[0][counter].x,2) + pow(this->RD[0][counter].y,2);
            RD_tot[1] += pow(this->RD[1][counter].x,2) + pow(this->RD[1][counter].y,2);
            RD_tot[2] += pow(this->RD[2][counter].x,2) + pow(this->RD[2][counter].y,2);


        }
        
        //Get next lines and increment counter
        getline(dipolesFile,lineDip);
        getline(positionsFile,linePos);
        counter++;

    }
    if(counter < this->N-1){
        cout << "ERROR. counter = " << counter << ", this->N = " << this->N;
        cout << ", in prepForAbsorb(). Aborting." << endl;
        return false;
    }


    //Finish mu_tot calculations
    mu_tot[0] = sqrt(mu_tot[0]);
    mu_tot[1] = sqrt(mu_tot[1]);
    mu_tot[2] = sqrt(mu_tot[2]);
    cout << endl << "mu_tot[0], mu_tot[1], mu_tot[2]:" << endl;
    cout << mu_tot[0] << "  " << mu_tot[1] << "  " << mu_tot[2] << endl;


    //Finish RD_tot calculations
    RD_tot[0] = sqrt(RD_tot[0]);
    RD_tot[1] = sqrt(RD_tot[1]);
    RD_tot[2] = sqrt(RD_tot[2]);
    cout << endl << "RD_tot[0], RD_tot[1], RD_tot[2]:" << endl;
    cout << RD_tot[1] << "  " << RD_tot[1] << "  " << RD_tot[2] << endl;


    //Don't need positions anymore
    delete[] this->positions[0];
    delete[] this->positions[1];
    delete[] this->positions[2];

    //Close files
    dipolesFile.close();
    positionsFile.close();


    return true;



}





//Prepares initial state for given direction
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
prepInitState(int dir) {

    cout << "Preparing initial state." << endl;

    //Absorption
    if(this->sysObjPtr->doingAbsorbSpec) {
        //Reset overall spectra counter
        this->timestepCounter = 1;

        tMat elementVal;

        cout << "Preparing state for absorb calc. State " << dir << ":" << endl;
        for(int site=0;site<this->N;site++) {

            elementVal.x = this->transitionDipoles[dir][site].x / mu_tot[dir];
            elementVal.y = this->transitionDipoles[dir][site].y / mu_tot[dir];
            this->sysObjPtr->matOfStates.setHostElement(site,0,elementVal);

        }

        this->sysObjPtr->matOfStates.copyHostColToDeviceCol(0);

        return true;

    //CD
    } else if (this->sysObjPtr->doingCDSpec) {


        this->timestepCounter = 1;

        tMat elementVal;

        cout << "Preparing state for CD calc. State " << dir << ":" << endl;
        for(int site=0;site<this->N;site++) {

            elementVal.x = this->RD[dir][site].x / RD_tot[dir];
            elementVal.y = this->RD[dir][site].y / RD_tot[dir];
            this->sysObjPtr->matOfStates.setHostElement(site,0,elementVal);

        }
        //Copy ket to device
        this->sysObjPtr->matOfStates.copyHostColToDeviceCol(0);


        cout << "Preparing 'bra' used for CD calc." << endl;
        for(int site=0;site<this->N;site++) {

            elementVal.x = this->transitionDipoles[dir][site].x / mu_tot[dir];
            elementVal.y = - this->transitionDipoles[dir][site].y / mu_tot[dir]; //negative because conjugate taken later
            this->h_braForCD[site] = elementVal;

        }

        //Copy bra to device
        this->d_braForCD = this->h_braForCD;

        return true;
    }


    //Should have exited function by this point
    cout << "You should not be in spectraClass->prepInitState()........" << endl;
    return false;

}




//Find M(t)=<psi(0)|psi> for one section of the propagation
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
calcCorrFunc(int stepsToDo) {

    cublasStatus_t cbStat;
    cout << "In calcCorrFunc(). stepsToDo = " << stepsToDo << endl << endl;

    //Doing dot product < psi(0) | psi(t) >
    //cublas?
    //for(int subStep=0; this->timestepCounter < (this->timestepCounter+stepsToDo) ; this->timestepCounter++) {
    for(int subStep=1; subStep < (stepsToDo + 1) ; subStep++) {
 

        if(this->timestepCounter > this->totalSteps) {
            cout << "ERROR in spectraClass::calcCorrFunc(). timestepCounter>totalSteps.";
            cout << "timestepCounter,totalSteps: " << this->timestepCounter << ",";
            cout << this->totalSteps << endl;
            cout << "subStep = " << subStep << endl << endl;
            return false;
        }


        //Bra used for inner product depends on whether doing abs or CD
        tMat *ptrBra;
        if(this->sysObjPtr->doingAbsorbSpec) {
            ptrBra = this->sysObjPtr->matOfStates.getDeviceColumnPtr(0); //psi(0)
        } else if(this->sysObjPtr->doingCDSpec) {
            ptrBra = thrust::raw_pointer_cast( &this->d_braForCD[0] );
        }


        // Manual: Notice that in the first equation the conjugate of the element 
        // of vector should be used if the function name
        // ends in character ‘c’.
        cbStat = cublasZdotc(   //result = x'*y
            this->sysObjPtr->cbHandle,  //handle
            this->N,    //n
            //this->sysObjPtr->matOfStates.getDeviceColumnPtr(0) ,            //x = psi(0)
            ptrBra,   //the bra
            1,          //stride
            this->sysObjPtr->matOfStates.getDeviceColumnPtr(subStep),   //y = psi(t)
            1,          //stride
            thrust::raw_pointer_cast(&this->d_timeCorrFunc[ this->timestepCounter ])            //result
        );


        if(cbStat!=CUBLAS_STATUS_SUCCESS) {
            cout << "ERROR running cublas in calcCorrFunc(). ";
            cout << cublasGetErrorString(cbStat) << "." << endl;
            cout << "this->timestepCounter: " << this->timestepCounter << ".";
            cout << " Aborting." << endl;
            return false;
        }


        this->timestepCounter ++;
        
    }


    return true;

}


//Take FFT and copy back to host:
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
singleSpectraCalc(int dir) {

    cufftResult_t cfErr;    //CUFFT_SUCCESS
    //cublasStatus_t cbErr;   //CUBLAS_STATUS_SUCCESS

    //Dress correlation function with gaussian
    if(this->lineShapeDressing) {

        //Parameter
        tReal sig = this->gaussLineshapeWidth;

        //Determine block and grid size
        int threadsPerBlock, numBlocks;
        threadsPerBlock = 128;
        numBlocks = int( ceil( float(this->totalSteps) / float(threadsPerBlock) ) );

        //Kernel for lineShapeDressing
        kernelGaussianLineDress<<<numBlocks,threadsPerBlock>>>(
            thrust::raw_pointer_cast(&this->d_timeCorrFunc[0]), //Function to dress, complex
            sig,        
            this->sysObjPtr->dt,         
            this->totalSteps  //So you know when to stop
        );

        //Check for error
        cudaError_t cuErr;
        cuErr = cudaGetLastError();
        if(cuErr!=cudaSuccess) {
            cout << "ERROR after kernelGaussianLineDress. ";
            cout << "Exiting." << endl;
            return false;
        }


    }


    //Create plan
    //cufftHandle plan;
    //cufftPlan1d(&plan, this->transformLength, CUFFT_Z2Z, 1);


    //Run FFT on the time-correlation function
    //cufftExecZ2Z(plan, idata, odata, direction[CUFFT_FORWARD, CUFFT_INVERSE])
    cfErr = cufftExecZ2Z(
        this->sysObjPtr->cufftPlan,
        //plan,
        thrust::raw_pointer_cast(&this->d_timeCorrFunc[0]), 
        thrust::raw_pointer_cast(&this->d_transformedFunc[0]), 
        CUFFT_INVERSE); //Should definitely be CUFFT_INVERSE
    if(cfErr!=CUFFT_SUCCESS) {
        cout << "ERROR running cufftExecZ2Z() in spectraClass::singleSpectraCalc(). ";
        cout << cufftGetErrorString(cfErr) << endl;
        cout << "Aborting." << endl;
        return false;
    }



    //Copy back M(t) and the transform
    this->h_timeCorrFunc[dir] = this->d_timeCorrFunc;
    this->h_transformedFunc[dir] = this->d_transformedFunc;

    //Will scale output by mu_tot[dir]^2
    float scalingConst;
    if(this->sysObjPtr->doingAbsorbSpec) {
        scalingConst = pow(mu_tot[dir],2);
    } else if(this->sysObjPtr->doingCDSpec) {
        float pi = 3.1415926535;
        scalingConst = (pi/2)*pow(10.,-7);
        scalingConst *= mu_tot[dir]*RD_tot[dir];
    }
    cout << "scalingConst = " << scalingConst << endl;

    //Open files
    ofstream corrFuncFile, specFile;
    if(this->sysObjPtr->doingAbsorbSpec) {
        corrFuncFile.open(this->absCorrFuncEachDir[dir].c_str());
        specFile.open(this->absFilenamesEachDir[dir].c_str());
    } else if(this->sysObjPtr->doingCDSpec) {
        corrFuncFile.open(this->CDCorrFuncEachDir[dir].c_str());
        specFile.open(this->CDFilenamesEachDir[dir].c_str());
    }

    //For now, print them
    corrFuncFile << "{timestep #} {timestamp fs} {M(t) real imag}: " << endl;
    for(int i=0;i<this->transformLength;i++) {
        corrFuncFile << setw(6) << i;
        corrFuncFile << setw(14) << i * this->sysObjPtr->dt;
        this->h_timeCorrFunc[dir][i].x *= scalingConst;
        this->h_timeCorrFunc[dir][i].y *= scalingConst;
        corrFuncFile << setw(14) << this->h_timeCorrFunc[dir][i].x;
        corrFuncFile << setw(14) << this->h_timeCorrFunc[dir][i].y;// << " *i";
        corrFuncFile << endl;
    }
    //cout << endl << "{k #} {freq cm^-1} {Transform}: " << endl;
    specFile << "{k #} {freq rev/fs} {freq cm^-1} {Transform real imag}: " << endl;
    int k;
    for(int freqId=0;freqId<this->transformLength;freqId++) {
        if(freqId > this->totalSteps/2) {
            k = - this->totalSteps + freqId;
        } else {
            k = freqId;
        }
        specFile << setw(6) << k;
        specFile << setw(14) << (double)k / (this->totalSteps * this->sysObjPtr->dt);
        //cout << setw(14) << this->h_scaleTransform * this->invFsToInvCm * k;
        specFile << setw(14) << this->invFsToInvCm * k / ( this->sysObjPtr->dt * this->totalSteps );
        this->h_transformedFunc[dir][freqId].x *= scalingConst / this->totalSteps;
        this->h_transformedFunc[dir][freqId].y *= scalingConst / this->totalSteps;
        specFile << setw(14) << this->h_transformedFunc[dir][freqId].x;
        specFile << setw(14) << this->h_transformedFunc[dir][freqId].y;// << " *i";
        specFile << endl;
    }


    //Close files
    corrFuncFile.close();
    specFile.close();


    //Print to overall spectrum file if on final run
    if(dir==2) {

        //Open files
        if(this->sysObjPtr->doingAbsorbSpec) {
            corrFuncFile.open(this->absCorrFunc.c_str());
            specFile.open(this->absFilename.c_str());
        } else if(this->sysObjPtr->doingCDSpec) {
            corrFuncFile.open(this->CDCorrFunc.c_str());
            specFile.open(this->CDFilename.c_str());
        }

        corrFuncFile << "{timestep #} {timestamp fs} {M(t) real imag}: " << endl;
        for(int i=0;i<this->transformLength;i++) {
            corrFuncFile << setw(6) << i;
            corrFuncFile << setw(14) << i * this->sysObjPtr->dt;
            corrFuncFile << setw(14) << 
                (this->h_timeCorrFunc[0][i].x + this->h_timeCorrFunc[1][i].x + this->h_timeCorrFunc[2][i].x)/3;
            corrFuncFile << setw(14) <<
                (this->h_timeCorrFunc[0][i].y + this->h_timeCorrFunc[1][i].y + this->h_timeCorrFunc[2][i].y)/3;// << " *i";
            corrFuncFile << endl;
        }

        int k;
        for(int freqId=0;freqId<this->transformLength;freqId++) {
            if(freqId > this->totalSteps/2) {
                k = - this->totalSteps + freqId;
            } else {
                k = freqId;
            }
            specFile << setw(6) << k;
            specFile << setw(14) << (double)k / (this->totalSteps * this->sysObjPtr->dt);
            //cout << setw(14) << this->h_scaleTransform * this->invFsToInvCm * k;
            specFile << setw(14) << this->invFsToInvCm * k / ( this->sysObjPtr->dt * this->totalSteps );
            specFile << setw(14) <<
                (this->h_transformedFunc[0][freqId].x + this->h_transformedFunc[1][freqId].x + this->h_transformedFunc[2][freqId].x)/3;
            specFile << setw(14) <<
                (this->h_transformedFunc[0][freqId].y + this->h_transformedFunc[1][freqId].y + this->h_transformedFunc[2][freqId].y)/3;// << " *i";
            specFile << endl;
        }


        //Close files
        corrFuncFile.close();
        specFile.close();

    }



    return true;

}


//Average the spectra together:
template <typename tMat, typename tState, typename tReal>
bool spectraClass<tMat,tState,tReal>::
completeSpectraCalc() {


    return true;

}



















