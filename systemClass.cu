/*
systemClass.cu
Holds state, Hamiltonian, timestep, etc., and includes function for propagating


*/

#include <systemClass.h>




// //Constructor
// template <typename tMat, typename tState, typename tReal>
// systemClass<tMat,tState,tReal>::systemClass() {

//     //this->systemClass(SIM_TYPE_CONST_HAM);

// }



//Constructor
template <typename tMat, typename tState, typename tReal>
systemClass<tMat,tState,tReal>::
systemClass() {


    // Set defaults
    this->simType = SIM_TYPE_CONST_HAM;
    this->propMode = PROP_MODE_LANCZOS;

    this->stateAllocated = false;
    this->N = -1;
    this->initialState = 0;
    this->boolInitFromFile=false;

    this->totalLoops = 1;

    //Debugging flags
    this->debugDiags = false;

    //Default ensemble average mode
    this->ensembleAvgMode = ENS_AVG_MODE_NONE;
    this->doStateLooping = false;

    //Printing populations
    this->printAllPops = false;
    this->writingOutPops = false;
    this->writingOutStates = false;
    this->writeBinaryPops = false;

    this->doingAbsorbSpec = false;
    this->doingCDSpec = false;
    // this->currentlyDoingAbsorb = false;
    // this->currentlyDoingCD = false;

    this->temperature = -1.;

    this->includeStaticDisorder = false;

    this->outFolderName = "outFolderDefault";

    this->totalAccumulatedError = 0.;

    //Printing participation ratio
    this->printPartic = false;

    //Defaults for Taylor-Krylov
    this->pTaylorKrylov = 10;
    this->mMatBTaylorKrylov = 24;

    //Allocate for timstep variable
    cudaMalloc( (void**)&d_dt, sizeof(tMat));

    //Allocate for state norm
    cudaMalloc( (void**)&d_stateNorm, sizeof(tReal));
    cudaMalloc( (void**)&d_stateInvNorm, sizeof(tReal));

    //Declare helper objects


    

}


//Set cusparse handle
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::setCusparseHandle(cusparseHandle_t *csHandle){

    this->csHandle = *csHandle;

    this->h_hamCoo.setCusparseHandle(csHandle);
    this->d_hamCoo.setCusparseHandle(csHandle);
    this->d_hamCsr.setCusparseHandle(csHandle);

}


//Set cublas handle
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::setCublasHandle(cublasHandle_t *cbHandle){

    (this->cbHandle) = *cbHandle;

    // this->h_hamCoo.setCublasHandle(cbHandle);
    // this->d_hamCoo.setCublasHandle(cbHandle);
    // this->d_hamCsr.setCublasHandle(cbHandle);

}



//Read in Hamiltonian in COO format
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
readInHam() {

    //Read in matrix file
    cout << hamFileName << endl;
    bool requireAllDiagElems = false;
    if(this->simType != SIM_TYPE_CONST_HAM) requireAllDiagElems = true;
    if( ! this->h_hamCoo.readInFile(this->hamFileName, requireAllDiagElems) ) return false;

    //Set state length
    this->N = h_hamCoo.N;

    //Include static disorder if selected
    // if(this->includeStaticDisorder) {

    //     cout << "Including static disorder with sigma = " << this->sigmaStaticDisorder << endl;

    //     //Random number generator setup
    //     boost::mt19937 rng(clock()+time(0));
    //     // boost::normal_distribution<> nd(0.0, 1.0);
    //     boost::normal_distribution<> nd(0.0, this->sigmaStaticDisorder);
    //     boost::variate_generator<boost::mt19937&, 
    //                         boost::normal_distribution<> > var_nor(rng, nd);

    //     //= var_nor()

    //     //This will work correctly only if 'requireAllDiagElems' is set to true.
    //     for(int elem = 0; elem<h_hamCoo.nnz; elem++ ) {
    //         if( h_hamCoo.cooRowIndA[elem] == h_hamCoo.cooColIndA[elem] ) {

    //             tReal randvar = var_nor();
    //             this->h_hamCoo.cooValA[elem].x += randvar;
    //             //Print to test
    //             // cout << "diag# randval resultval: ";
    //             // cout << h_hamCoo.cooRowIndA[elem] << " " << randvar << " " << h_hamCoo.cooValA[elem].x;
    //             // cout << " " << h_hamCoo.cooValA[elem].y << " i" << endl;

    //         }
    //     }

    // }



    //Set up Hamiltonian on system
    if( ! this->d_hamCoo.createOnDevice(this->h_hamCoo) ) return false;

    //Copy to csr on device
    if( ! this->d_hamCsr.pointToCooAndConvert(this->d_hamCoo) ) return false;

    //Copy csr to host
    if( ! this->h_hamCsr.createFromDeviceCsrMat(this->d_hamCsr) ) return false;

    //Print to see that the conversion worked
    //this->h_hamCsr.printMatToScreen();



    return true;

}



//Method for reading in the input file
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
parseInFile(char inFileString[]) {

    //Declare
    //cudaError_t cudaErr;

    //cout << inFileString << endl;

    //Variables to be assigned to other objects later
    bool lineShapeDressing = false;
    tReal gaussLineshapeWidth = 1.0;

    //Declare and open filestream
    ifstream inFile;
    inFile.open( inFileString );
    if ( ! inFile.is_open() ) {
        cout << "ERROR. inFile failed to open. Aborting." << endl;
        return false;
    }

    //Set popsToPrint all to -1
    int arrlen = sizeof(this->popsToPrint)/sizeof(*(this->popsToPrint));
    cout << "arrlen = " << arrlen << endl;
    for(int i=0;i<arrlen;i++) {
        this->popsToPrint[i] = -1;
    }

    //Variables for reading
    //string line, strParam, strVal;

    while(inFile.good()) {

        //Variable for reading
        string line, strParam, strVal;

        getline(inFile, line);

        if( line.find("=")!=string::npos ) {

            stringSplit(line, strParam, strVal, '=');
            stringTrim(strParam);
            stringTrim(strVal);
            if( strParam.find("!")!=string::npos  ){
                continue;
            }



            if(strParam=="dt") {

                cout << strParam << " " << strVal << endl;
                this->dt = atof(strVal.c_str());

            } else if (strParam=="temperature") {

                cout << strParam << strVal << endl;
                this->temperature = atof(strVal.c_str());

            } else if (strParam=="debug_diags") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->debugDiags = false;
                } else if (val==1) {
                    this->debugDiags = true;
                } else {
                    cout << "ERROR. debugDiags must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="printpop0") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[0] = atoi(strVal.c_str());
            } else if (strParam=="printpop1") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[1] = atoi(strVal.c_str());
            } else if (strParam=="printpop2") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[2] = atoi(strVal.c_str());
            } else if (strParam=="printpop3") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[3] = atoi(strVal.c_str());
            } else if (strParam=="printpop4") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[4] = atoi(strVal.c_str());
            } else if (strParam=="printpop5") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[5] = atoi(strVal.c_str());
            } else if (strParam=="printpop6") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[6] = atoi(strVal.c_str());
            } else if (strParam=="printpop7") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[7] = atoi(strVal.c_str());
            } else if (strParam=="printpop8") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[8] = atoi(strVal.c_str());
            } else if (strParam=="printpop9") {
                cout << strParam << " " << strVal << endl;
                this->popsToPrint[9] = atoi(strVal.c_str());

            } else if (strParam=="print_partic") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->printPartic = false;
                } else if (val==1) {
                    this->printPartic = true;
                } else {
                    cout << "ERROR. print_partic must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="totalsteps") {

                cout << strParam << " " << strVal << endl;
                //Note the plus-1, since 0-vector is initial vector
                this->totalSteps = atoi(strVal.c_str());

            } else if(strParam=="ensembleavgmode") {

                // cout << strParam << " " << strVal << endl;
                // int val = atoi(strVal.c_str());
                // if(val==0) {
                //     this->ensembleModeAverageOnly = false;
                // } else if (val==1) {
                //     this->ensembleModeAverageOnly = true;
                // } else {
                //     cout << "ERROR. ensemblemode must be 0 or 1. Aborting.";
                //     cout << endl;
                //     return false;
                // }

                cout << strParam << " " << strVal << endl;

                if(strVal=="ENS_AVG_MODE_NONE") {
                    this->ensembleAvgMode = ENS_AVG_MODE_NONE;
                } else if (strVal=="ENS_AVG_MODE_FIRST") {
                    this->ensembleAvgMode = ENS_AVG_MODE_FIRST;
                } else if (strVal=="ENS_AVG_MODE_CONTINUATION") {
                    this->ensembleAvgMode = ENS_AVG_MODE_CONTINUATION;
                } else {
                    cout << "this->ensembleAvgMode set to default, ";
                    cout << "ENS_AVG_MODE_NONE." << endl;
                    this->ensembleAvgMode = ENS_AVG_MODE_NONE;
                }


            } else if (strParam=="ensembleruns") {

                cout << strParam << " " << strVal << endl;
                this->numEnsembleRuns = atoi(strVal.c_str());

            } else if (strParam=="m") {

                cout << strParam << " " << strVal << endl;
                this->m = atoi(strVal.c_str());

            } else if (strParam=="ka_substeps") {

                cout << strParam << " " << strVal << endl;
                this->ka_subSteps = atoi(strVal.c_str());

            } else if (strParam=="dostatelooping") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->doStateLooping = false;
                } else if (val==1) {
                    this->doStateLooping = true;
                } else {
                    cout << "ERROR. dostatelooping must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="writeoutpops") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->writingOutPops = false;
                } else if (val==1) {
                    this->writingOutPops = true;
                } else {
                    cout << "ERROR. writeoutpops must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="writebinarypops") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->writeBinaryPops = false;
                } else if (val==1) {
                    this->writeBinaryPops = true;
                } else {
                    cout << "ERROR. writebinarypops must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }


            } else if (strParam=="printallpops") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->printAllPops = false;
                } else if (val==1) {
                    this->printAllPops = true;
                } else {
                    cout << "ERROR. printallpops must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }


            } else if (strParam=="writeoutstates") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                // cout << "***** " << val << endl;
                if(val<=0) {
                    this->writingOutStates = false;
                    this->writeStateFreq = val;
                } else if (val>=1) {
                    this->writingOutStates = true;
                    // cout << "***** " << val << endl;
                    this->writeStateFreq = val;
                    // cout << "***** " << this->writeStateFreq << endl;
                    // cout << "***** " << val << endl;
                    // cout << "***** " << this->writeStateFreq << endl;
                } else {
                    cout << "ERROR. writeoutstates must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="stepsperloop") {

                cout << strParam << " " << strVal << endl;
                this->stepsPerLoop = atoi(strVal.c_str());

            } else if (strParam=="outfolder") {

                cout << strParam << " " << strVal << endl;
                this->outFolderName = strVal;


            } else if (strParam=="hamfile") {

                cout << strParam << " " << strVal << endl;
                this->hamFileName = strVal;

            } else if (strParam=="statefile") {

                cout << strParam << " " << strVal << endl;
                this->stateFileName = strVal;

            } else if (strParam=="outfile") {

                cout << strParam << " " << strVal << endl;
                this->outFileName = strVal;

            } else if (strParam=="tkmatfile") {

                cout << strParam << " " << strVal << endl;
                this->tkMatFileName = strVal;

            } else if (strParam=="tk_p") {

                cout << strParam << " " << strVal << endl;
                this->pTaylorKrylov = atoi(strVal.c_str());

            } else if (strParam=="tk_m_aux") {

                cout << strParam << " " << strVal << endl;
                this->mMatBTaylorKrylov = atoi(strVal.c_str());

            } else if (strParam=="simtype") {

                cout << strParam << " " << strVal << endl;

                if(strVal=="SIM_TYPE_CONST_HAM") {
                    this->simType = SIM_TYPE_CONST_HAM;
                } else if (strVal=="SIM_TYPE_HSR") {
                    this->simType = SIM_TYPE_HSR;
                } else if (strVal=="SIM_TYPE_KUBO_ANDERSON") {
                    this->simType = SIM_TYPE_KUBO_ANDERSON;
                } else if (strVal=="SIM_TYPE_ZZ_REAL_NOISE") {
                    this->simType = SIM_TYPE_ZZ_REAL_NOISE;
                } else if (strVal=="SIM_TYPE_ZZ_COMPLEX_NOISE") {
                    this->simType = SIM_TYPE_ZZ_COMPLEX_NOISE;
                } else {
                    cout << "this->simType set to default, ";
                    cout << "SIM_TYPE_CONST_HAM." << endl;
                    this->simType = SIM_TYPE_CONST_HAM;
                }

            } else if (strParam=="propmode") {

                cout << strParam << " " << strVal << endl;

                if(strVal=="PROP_MODE_LANCZOS") {
                    this->propMode = PROP_MODE_LANCZOS;
                } else if (strVal=="PROP_MODE_TAYLOR_LANCZOS") {
                    this->propMode = PROP_MODE_TAYLOR_LANCZOS;
                } else {
                    cout << "this->propMode set to default, ";
                    cout << "PROP_MODE_LANCZOS." << endl;
                    this->propMode = PROP_MODE_LANCZOS;
                }

            } else if (strParam=="staticdis") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->includeStaticDisorder = false;
                } else if (val==1) {
                    this->includeStaticDisorder = true;
                } else {
                    cout << "ERROR. staticdis must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="sigmastatdis") {

                cout << strParam << " " << strVal << endl;
                this->sigmaStaticDisorder = atof(strVal.c_str());


            } else if (strParam=="hsr_stddevfile") {

                cout << strParam << " " << strVal << endl;
                this->hsr_filenameStdDev = strVal;
            
            } else if (strParam=="ka_paramsfile") {

                cout << strParam << " " << strVal << endl;
                this->ka_filenameStochParams = strVal;

            } else if (strParam=="zz_specd_file") {

                cout << strParam << " " << strVal << endl;
                this->zz_filenameSpecDens = strVal;

            } else if (strParam=="absorbspec") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->doingAbsorbSpec = false;
                } else if (val==1) {
                    this->doingAbsorbSpec = true;
                } else {
                    cout << "ERROR. absorbspec must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

                if(this->doingAbsorbSpec && this->doingCDSpec) {
                    cout << "ERROR. Cannot do both abs and CD spec. Aborting." << endl;
                    return false;
                }

            } else if (strParam=="cdspec") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->doingCDSpec = false;
                } else if (val==1) {
                    this->doingCDSpec = true;
                } else {
                    cout << "ERROR. cdspec must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

                if(this->doingAbsorbSpec && this->doingCDSpec) {
                    cout << "ERROR. Cannot do both abs and CD spec. Aborting." << endl;
                    return false;
                }

            } else if (strParam=="lineshapedress") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    lineShapeDressing = false;
                } else if (val==1) {
                    lineShapeDressing = true;
                } else {
                    cout << "ERROR lineshapedresss must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="gausswidth") {
            
                cout << strParam << " " << strVal << endl;
                gaussLineshapeWidth = atof(strVal.c_str());

            } else if (strParam=="dipolefile") {

                cout << strParam << " " << strVal << endl;
                this->filenameDipoleData = strVal;

            } else if (strParam=="posfile") {

                cout << strParam << " " << strVal << endl;
                this->filenamePosData = strVal;

            } else if (strParam=="initstate") {

                cout << strParam << " " << strVal << endl;
                this->initialState = atoi(strVal.c_str());


            } else if (strParam=="initfromfile") {

                cout << strParam << " " << strVal << endl;
                int val = atoi(strVal.c_str());
                if(val==0) {
                    this->boolInitFromFile = false;
                } else if (val==1) {
                    this->boolInitFromFile = true;
                } else {
                    cout << "ERROR. initfromfile must be 0 or 1. Aborting.";
                    cout << endl;
                    return false;
                }

            } else if (strParam=="initstatefile") {

                cout << strParam << " " << strVal << endl;
                this->initStateFileName = strVal;

            } else {
                //default
            } //End else-if train

        } //end if("=") statement

        //stringSplit(line, strParam, strVal, '=');

        //cout << line << endl;
        // cout << strParam << "  ";
        // cout << strVal << endl;


    } //End while loop


    //Deal with debug flags
    if(this->debugDiags) {
        fDiagDebugs.open("diags.debug.out");
    }

    //Create the out directory (http://stackoverflow.com/questions/7430248/how-can-i-create-new-folder-using-c-language)
    struct stat st = {0};
    if (stat( this->outFolderName.c_str() , &st) == -1) {
        mkdir(this->outFolderName.c_str(), 0700);
    }

    //Read in Hamiltonian Matrix
    if( ! this->readInHam() ) return false;

    //Copy timestep over
    cudaMemcpy(d_dt,&dt,sizeof(tReal),cudaMemcpyHostToDevice);

    //If printing out populations, initialize popPrint object
    if(this->writingOutPops) {
        this->popPrintObjPtr = new popPrintClass<tMat,tState,tReal>(this);
    }

    //If printing out states, initialize statePrint object
    if(this->writingOutStates) {
        cout << "Initializing State Printing..." << endl;
        this->statePrintObjPtr = new statePrintClass<tMat,tState,tReal>(this);
    }

    //If appopriate, read in HSR standard deviations
    if(this->simType == SIM_TYPE_HSR) {
        if(! this->prepForHsr() ) return false;
    }

    //Or read in KA stochastic parameters (relaxation time and stddevs)
    if(this->simType == SIM_TYPE_KUBO_ANDERSON) {
        if(! this->prepForKA() ) return false;
    }

    //Prep for ZZ_REAL if specified
    if(this->simType == SIM_TYPE_ZZ_REAL_NOISE) {
        if(! this->prepForZZReal() ) return false;
    }

    //Set up memory for static disorder
    if(this->includeStaticDisorder) {
        if( ! this->setupStaticDisorderMemory() ) return false;
    }

    //Prep for absorption spectra if chosen
    if(this->doingAbsorbSpec) {
        this->specObjPtr = new spectraClass<tMat,tState,tReal>(this);
        if(! this->specObjPtr->prepForAbsorb(this->filenameDipoleData) ) return false;
        this->specObjPtr->lineShapeDressing = lineShapeDressing;
        this->specObjPtr->gaussLineshapeWidth = gaussLineshapeWidth;
    }

    //Prep for CD spectra if chosen
    if(this->doingCDSpec) {
        this->specObjPtr = new spectraClass<tMat,tState,tReal>(this);
        if(! this->specObjPtr->prepForCD(this->filenameDipoleData, this->filenamePosData) ) return false;
        this->specObjPtr->lineShapeDressing = lineShapeDressing;
        this->specObjPtr->gaussLineshapeWidth = gaussLineshapeWidth;
    }


    //Prep cufft if doing spectra
    if(this->doingAbsorbSpec || this->doingCDSpec) {
        if(! initCufft( &this->cufftPlan, this->specObjPtr->transformLength /*+ 1*/ ) ) return false;
    }

    // Prep for participation ratio if appropriate
    if(this->printPartic) {
        cout << "Calculating and printing (pseudo-) inverse participation ratios." << endl << endl;
        // this->d_invParticRatio.resize(this->totalSteps);
        this->h_invParticRatio.resize(this->totalSteps);
        thrust::fill(this->h_invParticRatio.begin(), this->h_invParticRatio.end(), 0.);
    }

    //Clear up host memory (I think including static disorder shouldn't be affected.)
    if(! this->debugDiags) {
        this->h_hamCoo.freeMemory();
        this->h_hamCsr.freeMemory();
    }

    inFile.close();

    return true;

}


//Setup memory for the static disorder
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
setupStaticDisorderMemory() {

    cout << "Setting up memory for static disorder." << endl;

    //Initialize host array and store the original diagonal values
    this->h_origDiagValsFromFile.resize(this->N);
    for(int elem=0; elem < this->N; elem++) {
        this->h_origDiagValsFromFile[elem].x = this->h_diagInitVals[elem].x;
        this->h_origDiagValsFromFile[elem].y = this->h_diagInitVals[elem].y;
    }

    return true;

}



//Reset the static disorder
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
resetStaticDisorder() {

    //Populate the array with static disorder
    cout << "Including static disorder with sigma = " << this->sigmaStaticDisorder << endl;

    //Random number generator setup
    boost::mt19937 rng(clock()+time(0));
    // boost::normal_distribution<> nd(0.0, 1.0);
    boost::normal_distribution<> nd(0.0, this->sigmaStaticDisorder);
    boost::variate_generator<boost::mt19937&, 
                        boost::normal_distribution<> > var_nor(rng, nd);

    for(int elem = 0; elem < this->N; elem++ ) {

        tReal randvar = var_nor();
        this->h_diagInitVals[elem].x = randvar;

        //Add in the original diagonal values
        this->h_diagInitVals[elem].x += this->h_origDiagValsFromFile[elem].x;
        this->h_diagInitVals[elem].y += this->h_origDiagValsFromFile[elem].y;

        //Print to test
        // cout << "h_diagInitVals ";
        // cout << this->h_diagInitVals[elem].x << endl;
        // cout << "h_origDiagValsFromFile ";
        // cout << this->h_origDiagValsFromFile[elem].x << endl;

    }

    //Copy over to device
    this->d_diagInitVals = this->h_diagInitVals;

    //Run kernel adding them to matrix
    //Technically necessary only if its a constant hamiltonian run
    if(this->simType == SIM_TYPE_CONST_HAM) {

        cout << "ERROR. Code is not yet written for simType==SIM_TYPE_CONST_HAM ";
        cout << "combined with static disorder. Aborting." << endl;
        return false;
    }

    return true;

}


//Certain things are common prep for all non-constant Hamiltonians
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepForNonConstHam() {


    //Allocate vectors on host
    this->h_diagIndices.resize(this->N);
    this->h_diagInitVals.resize(this->N);

    //Get indices and values for diagional
    int diagCounter=0;
    for(int elem=0; elem < this->h_hamCoo.nnz; elem++) {

        if(this->h_hamCoo.cooRowIndA[elem]>diagCounter) {
            cout << "ERROR in systemClass::prepForNonConstHam(). ";
            cout << "h_hamCoo.cooRowIndA[elem]>diagCounter." << endl;
            cout << "elem = " << elem << ", h_hamCoo.cooRowIndA[elem] = ";
            cout << h_hamCoo.cooRowIndA[elem] << ", diagCounter = " << diagCounter;
            cout << "." << endl;
            return false;
        }

        //Assign indices and values
        if(this->h_hamCoo.cooRowIndA[elem] == this->h_hamCoo.cooColIndA[elem]) {
            this->h_diagIndices[diagCounter] = elem;
            this->h_diagInitVals[diagCounter].x = h_hamCoo.cooValA[elem].x;
            this->h_diagInitVals[diagCounter].y = h_hamCoo.cooValA[elem].y;
            diagCounter++;
        }

    }
    if(diagCounter < this->N) {
        cout << "ERROR in systemClass::prepForNonConstHam(). ";
        cout << "After loop, diagCounter = " << diagCounter;
        cout << ", this->N = " << this->N;
        cout << "." << endl;
        return false;
    }

    //Test
    // cout << endl;
    // cout << "First and last few original-Hamiltonian diag indices:" << endl;
    // cout << h_diagIndices[0] << " " << h_diagIndices[1] << " ";
    // cout << h_diagIndices[2] << " " << h_diagIndices[3] << " ";
    // cout << h_diagIndices[this->N-4] << " " << h_diagIndices[this->N-3] << " ";
    // cout << h_diagIndices[this->N-2] << " " << h_diagIndices[this->N-1] << " ";
    // cout << endl << endl;

    //Test: Print all
    // cout << "diagIndices and diagInitVals" << endl;
    // for(int i=0;i<this->N;i++) {
    //     cout << h_diagIndices[i] << "  ";
    //     cout << h_diagInitVals[i].x << "  " << h_diagInitVals[i].y << "  ";
    // }cout << endl << endl;

    //Copy over diag indices and diag elements
    this->d_diagIndices = this->h_diagIndices;
    this->d_diagInitVals = this->h_diagInitVals;


    return true;

}


//Prepare cuRand
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepForCurand() {


    //Allocate for curand states
    this->blockSizeCurand = 64;
    this->numBlocksCurand = 64;
    this->numCurandStates = this->blockSizeCurand*this->numBlocksCurand; //72*16384 = 1.2MB
    this->sizeCurandStates = this->numCurandStates * sizeof(curandStateMRG32k3a);
    if( cudaMalloc((void **)&(this->devMRGStates),  this->sizeCurandStates) != cudaSuccess ) {
        cout << "ERROR allocating for devMRGStates in prepForCurand()." << endl;
        return false;
    }

    //Initialize Curand
    time_t seed;
    time(&seed);
    setupPRNG<<<this->numBlocksCurand,this->blockSizeCurand>>>
        (seed, this->devMRGStates);
    if(cudaGetLastError()!=cudaSuccess) {
        cout << "ERROR found after setupPRNG() in prepForCurand(). ";
        cout << "Error string: " << cudaGetErrorString(cudaGetLastError()) << endl;
        return false;
    }

    return true;




}



//Read in HSR Standard Deviations and prepare arrays
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepForHsr() {

    cout << "Prepping for HSR." << endl;

    //cudaError_t cuErr;

    //Do routine common to all non-constant hamiltonians
    if( ! this->prepForNonConstHam() ) return false;
    //Prep curand
    if( ! this->prepForCurand() ) return false;


    //Allocate vector on host
    this->h_hsr_stdDev.resize(this->N);


    //Open filestream
    ifstream inFile;
    inFile.open( this->hsr_filenameStdDev.c_str() );
    if ( ! inFile.is_open() ) {
        cout << "ERROR. HSR Parameter file failed to open. ";
        cout << "file: " << this->hsr_filenameStdDev << ". Aborting." << endl;
        return false;
    }


    //Variables for reading
    string line; //, strParam, strVal;

    int counter=0;
    while(inFile.good()) {

        getline(inFile, line);

        if(counter >= this->N) {
            cout << "WARNING in systemClass::prepForHsr. counter = " << counter << ", this-> N = ";
            cout << this->N << ". Ignoring and proceeding." << endl;
        } else {
            this->h_hsr_stdDev[counter] = atof(line.c_str());
        }

        counter++;

    }
    if(counter < this->N-1){
        cout << "ERROR. counter = " << counter << ", this->N = " << this->N;
        cout << ", in prepForHsr(). Aborting." << endl;
        return false;
    }



    //Close file
    inFile.close();



    //Copy stddev vector to device
    this->d_hsr_stdDev = this->h_hsr_stdDev;


    return true;

}



//Read in KA stochastic parameters and prepare arrays
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepForKA() {

    cout << "Preparing for Kubo-Anderson." << endl;

    //Do routine common to all non-constant hamiltonians
    if( ! prepForNonConstHam() ) return false;
    //Prep curand
    if( ! this->prepForCurand() ) return false;


    //Set substep for propagating Langevin equation
    this->ka_subdt = this->dt / (float)this->ka_subSteps;
    this->ka_sqrtSubdt = sqrt(this->ka_subdt);

    //Allocate vectors on host
    this->h_ka_dEps.resize(this->N);
    this->h_ka_invRelaxTimes.resize(this->N);
    this->h_ka_stochCoeffs.resize(this->N);


    //Open filestream
    ifstream inFile;
    inFile.open( this->ka_filenameStochParams.c_str() );
    if ( ! inFile.is_open() ) {
        cout << "ERROR. KA Parameter file failed to open. ";
        cout << "file: " << this->ka_filenameStochParams << ". Aborting." << endl;
        return false;
    }

    //Random number generator setup
    boost::mt19937 rng(clock()+time(0));
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, 
                               boost::normal_distribution<> > var_nor(rng, nd);


    //Variables for reading
    string line;//, strRelaxTime, strRootVariance;//, strParam, strVal;
    tReal relaxTime, stdDev;
    int counter=0;
    getline(inFile, line); //get first line
    while(inFile.good()) {

        if(counter >= this->N) {
            cout << "WARNING in systemClass::prepForKA(). counter = " << counter << ", this-> N = ";
            cout << this->N << ". Ignoring and proceeding." << endl;
        } else {
            stringstream ss;
            ss << line;
            ss >> relaxTime;
            ss >> stdDev;
            this->h_ka_invRelaxTimes[counter] = 1./relaxTime;
            this->h_ka_stochCoeffs[counter] = stdDev*sqrt(2./relaxTime);
            //Initialize dEps with gaussian noise
            this->h_ka_dEps[counter] = stdDev*var_nor();
            if(counter==0) {
                cout << "For counter==0, ";
                cout << "relaxTime = " << relaxTime << "     ";
                cout << "stdDev = " << stdDev << "     ";
                cout << "Random start for dEps: " << this->h_ka_dEps[counter] << endl;
            }

        }

        counter++;
        getline(inFile, line);

    }
    if(counter < this->N-1){
        cout << "ERROR. counter = " << counter << ", this->N = " << this->N;
        cout << ", in prepForKA(). Aborting." << endl;
        return false;
    }


    //Close file
    inFile.close();

    //Copy vectors to device
    this->d_ka_dEps = this->h_ka_dEps;
    this->d_ka_invRelaxTimes = this->h_ka_invRelaxTimes;
    this->d_ka_stochCoeffs = this->h_ka_stochCoeffs;


    //Populate initial Hamiltonian (to include initial dEps values)
    //if( ! this->updateHamKuboAnderson() ) return false;
    //Not needed becaus eit's already done!

    return true;
}


//Prepare for ZZReal-type Hamiltonian updating
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepForZZReal() {

    cout << "Preparing for ZZReal type simulation. " << endl;

    //Check that temperature is positive
    if( this->temperature <= 0 ) {
        cout << "ERROR in prepForZZReal. temperature must be greater than 0." << endl;
        cout << "this->temperature = " << this->temperature << endl;
        return false;
    }

    //Do routine common to all non-constant hamiltonians
    if( ! this->prepForNonConstHam() ) return false;

    //Open filestream
    ifstream inFile;
    inFile.open( this->zz_filenameSpecDens.c_str() );
    if ( ! inFile.is_open() ) {
        cout << "ERROR. Spectral Density file failed to open. ";
        cout << "file: " << this->zz_filenameSpecDens << ". Aborting." << endl;
        return false;
    }

    //Variables for reading
    string line;
    //Get first line, which is: {dOmega} {numOscillators}
    getline(inFile, line);
    stringstream ss;
    ss << line;
    ss >> this->dOmega;
    ss >> this->numZZoscPerSite;

    //Declare arrays
    this->numZZoscAllSites = this->N * this->numZZoscPerSite;
    thrust::host_vector<tReal> h_zzOmegaVals(this->numZZoscPerSite);
    thrust::host_vector<tReal> h_zzFlucCoeffs(this->numZZoscPerSite);


    int oscCounter=0;
    tReal omega, coeffVal, jVal, G_of_omeg;
    tReal hbar = 1.0;
    getline(inFile, line); //get first line
    while(inFile.good()) {

        if(oscCounter >= this->numZZoscPerSite) {
            cout << "WARNING in systemClass::prepForZZReal(). oscCounter = " << oscCounter;
            cout << ", this->numZZoscPerSite = ";
            cout << this->numZZoscPerSite << ". Ignoring and proceeding." << endl;
        } else {
            stringstream ss;
            ss << line;
            ss >> omega; //in wavenumbers
            ss >> jVal;
            //CHECK THIS. NOTES JUNE 3 2014.
            G_of_omeg = jVal / (1 - exp( -1./(BOLTZ_INVCM*this->temperature) * hbar * omega) );
            coeffVal = sqrt(2) * sqrt(G_of_omeg * this->dOmega / PI);
            // for(int site=0;site<this->N;site++) {
            //     int index = this->N * oscCounter + site;
            //     h_zzOmegaVals[index] = omega / INVFS_TO_INVCM; //Convert 1/cm to 1/fs
            //     h_zzFlucCoeffs[index] = coeffVal;
            // }
            h_zzOmegaVals[oscCounter] = omega / INVFS_TO_INVCM; //Convert 1/cm to 1/fs
            h_zzFlucCoeffs[oscCounter] = coeffVal;
            
            if(oscCounter==0) {
                cout << "For counter==0, ";
                cout << "jVal = " << jVal << "    ";
                cout << "this->temperature = " << this->temperature << endl;
                cout << "OmegaVal = " << h_zzOmegaVals[oscCounter] << "     ";
                cout << "G_of_omeg = " << G_of_omeg << "     ";
                cout << "coeffVal = " << coeffVal << endl;
                // cout << "1./(BOLTZ_INVCM*this->temperature) * HBAR_INVCM * omega = ";
                // cout << 1./(BOLTZ_INVCM*this->temperature) * HBAR_INVCM * omega;
                cout << endl << endl;
            }

        }

        oscCounter++;
        getline(inFile, line);

    }
    if(oscCounter < this->numZZoscPerSite){
        cout << "ERROR. oscCounter = " << oscCounter;
        cout << ", this->numZZoscPerSite = " << this->numZZoscAllSites;
        cout << ", in prepForZZReal(). Aborting." << endl;
        return false;
    }

    //Populate the rest of the vectors


    //Copy vectors to device (should prob be const memory)
    this->d_zzOmegaVals = h_zzOmegaVals;
    this->d_zzFlucCoeffs = h_zzFlucCoeffs;


    //Populate a vector random numbers
    thrust::host_vector<tReal> h_phi_n;
    h_phi_n.resize(this->numZZoscAllSites);

    //Use Boost to fill with rand uniform distr (0,2*pi)
    //Random number generator setup
    boost::mt19937 rng(clock()+time(0));
    boost::uniform_real<> ud(0.0, 2*PI);
    boost::variate_generator<boost::mt19937&, 
                               boost::uniform_real<> > gen_unif(rng, ud);

    for( int phiIndex = 0; phiIndex<h_phi_n.size(); phiIndex++ ) {
        h_phi_n[phiIndex] = gen_unif();
    }

    //Declare phi_n values on device
    this->d_phi_n = h_phi_n;

    //Print to test random host numbers
    // cout << "index, h_phi_n, h_zzOmegaVals, h_zzFlucCoeffs: " << endl;
    // for(int i=0;i<this->numZZoscPerSite;i++) {
    //     cout << i << " " << h_phi_n[i*this->N];
    //     // cout << " " << h_zzOmegaVals[i*this->N] << " ";
    //     // cout << h_zzFlucCoeffs[i*this->N];
    //     cout << endl;
    // } cout << endl;

    //Populate the first Hamiltonian
    // updateHamZZReal(0.0);
    //Don't need to because it's updated before the first propagation

    return true;

}


//Prepare for ZZReal-type Hamiltonian updating
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
zzRealResetRandomPhases() {

    cout << "Resetting random phases for ZZReal." << endl;

    //Populate a vector random numbers
    thrust::host_vector<tReal> h_phi_n;
    h_phi_n.resize(this->numZZoscAllSites);

    //Use Boost to fill with rand uniform distr (0,2*pi)
    //Random number generator setup
    //boost::mt19937 rng(time(0));
    boost::mt19937 rng(clock()+time(0));
    boost::uniform_real<> ud(0.0, 2*PI);
    boost::variate_generator<boost::mt19937&, 
                               boost::uniform_real<> > gen_unif(rng, ud);

    for( int phiIndex = 0; phiIndex<h_phi_n.size(); phiIndex++ ) {
        h_phi_n[phiIndex] = gen_unif();
    }

    //Copy phi_n values to device
    this->d_phi_n = h_phi_n;

    //For testing
    // cout << "First few d_phi_n values: " << endl;
    // cout << this->d_phi_n[0] << "  " << this->d_phi_n[1] << "  ";
    // cout << this->d_phi_n[2] << "  " << this->d_phi_n[3] << endl;

    return true;

}




// //Repopulate real nosie array
// template <typename tMat, typename tState, typename tReal>
// bool systemClass<tMat,tState,tReal>::
// repopulateRealNoise() {

//     cout << "Repopulating noise array." << endl;

//     if( curandGenerateNormal(
//             this->crGen,
//             //thrust::raw_pointer_cast(&this->d_realNoiseArray[0]),
//             this->d_realNoiseArray,
//             //sizeof(tReal)*this->d_realNoiseArray.size(),
//             this->realNoiseSize,
//             0., //mean
//             1.  //stddev
//         ) != CURAND_STATUS_SUCCESS) {
//         cout << "ERROR in curandGenerateNormalDouble() in repopulateRealNoise().";
//         cout << " Aborting." << endl;
//         return false;
//     } 

//     cout << "Done repopulation noise array." << endl;
//     //Print to test


//     return true;
// }



//Allocate memory on host and device for state vector
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
allocateState() {

    if(this->N <= 0) {
        cout << "ERROR in allocateState(). this->N = ";
        cout << this->N << endl;
        return false;
    }

    //Set memory size for state
    this->stateMemSize = this->N * sizeof(tState);

    //Allocate on host
    cout << "this->totalSteps = " << this->totalSteps << endl;
    if(this->doStateLooping) {

        //Set size of matrix to Nx(stepsPerLoop+1)
        //*** It's plus-one because initial state is kept there whole time ***
        cout << "Doing state looping. Steps per loop = " << this->stepsPerLoop << endl;
        this->matOfStates.allocateOnHostAndDevice(this->N,this->stepsPerLoop+1);

        //Set variables for use later
        this->totalLoops = this->getNumTotalLoops();
        this->stepsInLastLoop = this->getNumStepsThisLoop( 
                                (this->stepsPerLoop)*(this->totalLoops - 1) );
        cout << "totalLoops, stepsInLastLoop: " << this->totalLoops << ",";
        cout << this->stepsInLastLoop << endl << endl;

    } else {
        //Set size of matrix to Nx(totalsteps+1)
        this->matOfStates.allocateOnHostAndDevice(this->N,this->totalSteps+1);
    }


    //Set flag
    this->stateAllocated = true;

    return true;

}




//Initialize state based on 
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
setInitState() {

    if(this->boolInitFromFile) {
        //Initialize from file
        if( !this->initStateFromFile(this->initStateFileName) ) return false;
    } else {
        //Initialize single site from infile parameters
        if( !this->initStateAsBasisVector(this->initialState) ) return false;
    }

    //Copy the column to the device within the matrix object
    this->matOfStates.copyHostColToDeviceCol(0);

    return true;

}


//Initialize state as a unit basis vector
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
initStateAsBasisVector(int basisElement) {

    cout << "Beginning initStateAsBasisVector()." << endl;
    cout << "Setting intial state to " << basisElement << "." << endl;

    //Allocate state if it's not already allocated
    if(! this->stateAllocated) {
        cout << "Allocating state." << endl;
        if(! this->allocateState())
            return false;
    }

    //Make sure basisElement isn't larger than state size
    if(basisElement>this->N) {
        cout << "ERROR in initStateAsBasisVector. basisElement = ";
        cout << basisElement << ", this->N = " << this->N << endl;
        return false;
    }

    //Set the host vector appropriately
    tState oneReal; oneReal.x = 1.; oneReal.y = 0.;
    tState zero; zero.x = 0.; zero.y = 0.;
    for(int i=0;i<this->N;i++){
        this->matOfStates.setHostElement(i, 0, zero);
    }
    this->matOfStates.setHostElement(basisElement, 0, oneReal);



    return true;

}



template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
initStateFromFile(string& stateFileName) {

    cout << endl << "Initializing state from file. Normalizing internally." << endl << endl;

    //Allocate state if it's not already allocated
    if(! this->stateAllocated) {
        if(! this->allocateState())
            return false;
    }

    /*
    Format:
    BASE_{ZERO,ONE}
    {ind} {wavevector coeff}
    ...   ...
    END
    */

    //Open file
    ifstream inFile;
    inFile.open( stateFileName.c_str() );
    if (! inFile.is_open() ) {
        cout << "ERROR. State file '" << stateFileName << "' failed to open." << endl;
        return false;
    }


    string line;

    //Reading whether base zero or base one
    int baseIndexOffset;
    getline(inFile, line);
    if (line=="BASE_ZERO") {
        baseIndexOffset = 0;
    } else if (line=="BASE_ONE") {
        baseIndexOffset = 1;
    } else {
        cout << "ERROR. Must specify 'BASE_ZERO' or 'BASE_ONE' on first line ";
        cout << "of state file. Aborting." << endl;
        return false;
    }


    //Set the host vector to all zeros
    tState zero; zero.x = 0.; zero.y = 0.;
    for(int i=0;i<this->N;i++){
        this->matOfStates.setHostElement(i, 0, zero);
    }
    //this->matOfStates.setHostElement(basisElement, 0, oneReal);

    //Purge comment line
    getline(inFile, line);

    //Read states, until hit "END"
    int ind; float vecCoeff;
    while(getline(inFile, line)) {

        stringstream ss;
        tState val;

        //Check of end of file
        if( line.find("END") != std::string::npos ) break;

        //Get vals
        ss.str(line);
        ss >> ind;
        ss >> vecCoeff;
        val.x=vecCoeff;val.y=0;
        cout << "ind,vecC " << ind << "," << vecCoeff << endl;

        this->matOfStates.setHostElement(ind - baseIndexOffset, 0, val);

    }

    //Warn if last line isn't 'END'
    if( line.find("END") != std::string::npos ) {
        cout << endl << "WARNING. Initial state file did not end with 'END' keyword.";
        cout << endl << endl;
    }

    //Normalize state
    cout << "Normalizing initial state." << endl;
    this->matOfStates.normalizeHostCol(0);

    //Test: print initial state
    cout << "Initial state:" << endl;
    for(int i=0;i<matOfStates.numRows;i++) {
        cout << setw(10) << i << setw(20) << matOfStates.getHostElement(i,0).x;
        cout << " + i*" << setw(20) << matOfStates.getHostElement(i,0).y << endl;
    }

    //Close file
    inFile.close();

    return true;

}


//Initialize state as (1,1,1,...,1)/norm.
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
initStateAsUnitSpreadVector() {

    //Allocate state if it's not already allocated
    if(! this->stateAllocated) {
        if(! this->allocateState() )
            return false;
    }


    //Set the host vector appropriately
    tMat val; val.x = 1. / sqrt( (float)this->N); val.y=0;
    for(int i=0;i<this->N;i++){
        this->matOfStates.setHostElement(i, 0, val);
    }

    //Copy the column to the device within the matrix object
    this->matOfStates.copyHostColToDeviceCol(0);


    return true;

}


//Returns the number of initial conditions (e.g. for abs spec it's 3)
template <typename tMat, typename tState, typename tReal>
int systemClass<tMat,tState,tReal>::
getNumInitConditions() {

    int numInitConds = 0;

    //Can have things other than spectrums to consider later on
    if(this->doingAbsorbSpec || this->doingCDSpec) {
        numInitConds += 3;
    }

    //Default one run
    if(numInitConds==0) numInitConds = 1;

    //Return value
    return numInitConds;

}



//If necessary, prepare next initial state
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepNextInitState(int iterInitState) {



    //Used only if doing the absorption (other options available later)
    if(this->doingAbsorbSpec || this->doingCDSpec) {

        if(! this->specObjPtr->prepInitState(iterInitState) ) return false;

    }


    return true;
}

//Prep certain analyses before the run
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
prepAnalysesForNextRun() {

    //Reset the error
    this->totalAccumulatedError = 0.;

    //Deal with popPrintObj
    if(this->writingOutPops) {
        if( ! popPrintObjPtr->prepForNextRun() ) return false;
    }

    //Reset static disorder
    if(this->includeStaticDisorder) {

        //Test some values from previous hamiltonian to see if statdis being incorporated
        // cout << "First and last diagonal values from _previous_ Hamiltonian: " << endl;
        // tMat someDiagVals[6];
        // for(int i=0;i<3;i++) {
        //     int index = this->h_diagIndices[i];
        //     tMat* dPtr = &( this->d_hamCsr.csrValA[ index ] );
        //     cudaMemcpy(&someDiagVals[i], dPtr, sizeof(tMat), cudaMemcpyDeviceToHost);
        // }
        // cout << "this->N = " << this->N << endl;
        // for(int i=0;i<3;i++) {
        //     int index = this->h_diagIndices[this->N -3+i];
        //     cout << "index = " << index << endl;
        //     tMat* dPtr = &( this->d_hamCsr.csrValA[ index ] );
        //     cudaMemcpy(&someDiagVals[i+3], dPtr, sizeof(tMat), cudaMemcpyDeviceToHost);
        // }
        // for(int i=0;i<6;i++) {
        //     cout << someDiagVals[i].x << " + " << someDiagVals[i].y << "i, ";
        //     if(i==2) cout << " ... " << endl;
        // }


        //Reset static disorder
        if( ! this->resetStaticDisorder() ) return false;

    }

    //Re-randomize Hamiltonian if appropriate
    if(this->simType == SIM_TYPE_ZZ_REAL_NOISE) {
        if ( ! this->zzRealResetRandomPhases() ) return false;
    }
    if(this->simType==SIM_TYPE_HSR) {
        // not needed for hsr
    }
    if(this->simType==SIM_TYPE_KUBO_ANDERSON) {
        // not really needed for KA either. can start where left off.
    }



    return true;

}


//Run extra routines after single run as necessary, e.g. routines required for spectra
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
extraAnalysisAfterSingleLoop(int loopNum, int currentStep, int numStepsThisLoop) {

    // cudaError_t cuErr;



    //If doStateLooping is turned off then we don't worry about looping parameters
    int numRelevantTimesteps;
    if(this->doStateLooping) {
        if( loopNum == this->totalLoops - 1 ) {
            numRelevantTimesteps = this->stepsInLastLoop;
        } else {
            numRelevantTimesteps = this->stepsPerLoop;
        }
    } else {
        numRelevantTimesteps = this->totalSteps;
    }
    
    // If doing state printing, do it
    if(this->writingOutStates) {

        // Copy states back
        this->matOfStates.copyThisDeviceToThisHost();

        // Print states from this loop
        this->statePrintObjPtr->printStatesFromLoop(loopNum, currentStep, numStepsThisLoop);

    }



    //If keeping track of populations, copy populations over
    if(this->writingOutPops) {
        //Copy back state and write out the sites in popsToPrint[]
        //It's wasteful to copy back all the states though.....
        //This is where you should use paging I guess
        //The current scheme is kind of ridiculous



        int beginningStepNum = currentStep - this->stepsPerLoop;
        if(this->printAllPops) {

            //For debuggin
            cout << "numRows, numCols = " << this->matOfStates.numRows << ",";
            cout << this->matOfStates.numCols << endl;

            //Copy back entire matrix of states
            this->matOfStates.copyThisDeviceToThisHost();
            //Does synchronizing help?
            // cudaDeviceSynchronize();
            //Nope.

            //Print out populations
            if( !this->popPrintObjPtr->printOneLoopAllPops(beginningStepNum, numStepsThisLoop, this) ) return false;

        } else { //Print only specific populations

            //First copy back relevant states.

            //Code for this not written yet:
            // if(!this->popPrintObjPtr->printOneLoopSomePops()) return false;

        }



    }

    //If doing absorption spectra, calculate for this timeloop
    if(this->doingAbsorbSpec || this->doingCDSpec) {
        //Calculate one section of M(t)
        if( ! this->specObjPtr->calcCorrFunc(numRelevantTimesteps) ) return false;

    }

    return true;

}



//Run extra routines after single run as necessary, e.g. routines required for spectra
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
extraAnalysisAfterSingleRun(int run) {

    // //This function performed only if not in doingStateLooping mode
    // if( ! this->doStateLooping ) {
    //     //For absorption, Calculate M(t) = < psi0 | psi(t) >
    //     if(this->doingAbsorbSpec==1) {
    //

    //Handle things in population printer object
    if(this->writingOutPops) {
        if( ! popPrintObjPtr->doTasksForFinishedRun() ) return false;
    }

    return true;
}


//Run extra routines after full ensemble as necessary, e.g. routines required for spectra
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
extraAnalysisAfterEnsemble(int initialConditionNumber) {

    //I think this might mess up the flow of the program. Look into it.

    //If doing absorption spectra, complete the calculation for a single direction
    if( this->doingAbsorbSpec || this->doingCDSpec) {
        if( ! this->specObjPtr->singleSpectraCalc(initialConditionNumber) ) return false;
    }
    return true;

}


//Run extra routines after all init cond ensemble runs as necessary, e.g. routines required for spectra
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
extraAnalysisEnd() {

    // Inverse participation ratio
    if(this->printPartic) {
    
        ofstream outFile("invParticRat.dat");

        outFile << "# Inverse Particle Ratios" << endl;
        outFile << "# {timestep} {IPR}}" << endl;
        for(int i=0; i<this->h_invParticRatio.size(); i++) {
            // Note the renormalization by numEnsembleRuns
            outFile << i << "  " <<  this->h_invParticRatio[i] / this->numEnsembleRuns  << endl;
        }

        outFile.close();

    }
    
    // Close files for printing states, if appropriate
    if(this->doingAbsorbSpec || this->doingCDSpec) {

        this->statePrintObjPtr->closeFiles();

    }


    return true;
}





//Normalize state back to 1
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
renormalizeAndStoreNorm(int destVecNum) {


    // int stateNum = step + 1;
    // tReal hNorm;

    cublasStatus_t cbStatus;

    tMat* ptrState = this->matOfStates.getDeviceColumnPtr(destVecNum);


    //Calculate norm (result will be real scalar, placed in real part of your variable)
    cbStatus = cublasDznrm2(
        this->cbHandle,
        this->N,
        ptrState,
        1, //stride
        d_stateNorm
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR in systemClass::renormalizationStep(), cbulasZdscal().";
        cout << " Aborting." << endl;
        return false;
    }


    //Copied back because it is used for error analysis
    cudaMemcpy( &(this->h_norm), d_stateNorm, sizeof(tReal), cudaMemcpyDeviceToHost );

    //Print out renormalzation
    // cout << "After calculation for loop's state " << destVecNum << ",";
    // cout << "state norm = " << hNorm << ". Renormalizing now." << endl << endl;


    //Invert the norm
    //invertSingle<<<>>>(typeMat* destArr, typeMat* sourceArr, int ind);
    invertSingle<<<1,1>>>(d_stateInvNorm, d_stateNorm, 0);



    //Scale back
    cbStatus = cublasZdscal(
            this->cbHandle,
            this->N,
            &d_stateInvNorm[0], //alpha
            ptrState, //array pointer
            1 //stride
    );
    if(cbStatus!=CUBLAS_STATUS_SUCCESS) {
        cout << "ERROR in systemClass::renormalizationStep(), cbulasZdscal().";
        cout << " Aborting." << endl;
        return false;
    }




    return true;
}




//Copy state back to host
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
copyAllStatesBackToHost(){

    //Error if states have not been allocated
    if(! this->stateAllocated) {
        cout << "ERROR in copyStateBackToHost(). stateAllocated is false.";
        cout << endl;
        return false;
    }

    //Copy to host
    if( ! this->matOfStates.copyThisDeviceToThisHost() ) return false;

    return true;

}







//Get pointer to first element of a given state column
template <typename tMat, typename tState, typename tReal>
tState* systemClass<tMat,tState,tReal>::
getStateDevPtr(int stateNum) {

    return this->matOfStates.getDeviceColumnPtr(stateNum);

}


//Copy CSR Hamiltonian back to host. Used for testing.
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
copyCsrHamDeviceToHost(){

    this->h_hamCsr.copyFromDeviceCsrMat(this->d_hamCsr);

    return true;
}


//Update Hamiltonian for next timestep
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
updateHamiltonian(tReal timeVal) {

    if(this->simType == SIM_TYPE_HSR) {
        // cout << "Updating HSR Hamiltonian." << endl << endl;
        if( ! this->updateHamHakenStrobl() ) return false;
    } else if(this->simType == SIM_TYPE_KUBO_ANDERSON) {
        //cout << "Updating KA Hamiltonian." << endl << endl;
        if( ! this->updateHamKuboAnderson() ) return false;
    } else if(this->simType == SIM_TYPE_ZZ_REAL_NOISE) {
        //cout << "Updating ZZ-real Hamiltonian" << endl << endl;
        if( ! this->updateHamZZReal(timeVal) ) return false;
    }

    return true;
}



//Update Hamiltonian for HSR
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
updateHamHakenStrobl() {

    // if( this->noiseElemCounter + this->N >= this->realNoiseLength ) {
    //     this->repopulateRealNoise();
    //     this->noiseElemCounter = 0;
    // } else {
    //     this->noiseElemCounter = this->noiseElemCounter + this->N;
    // }


    //Put this blockSize managment earlier and store for whole object
    int threadsPerBlock = 128;
    int numBlocks = (int) ceil( (double)this->N / (double)threadsPerBlock );
    addDiagToRealGaussNoise<<<numBlocks,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&this->d_hamCsr.csrValA[0]), // typeMat *HamVals,
        thrust::raw_pointer_cast(&this->d_diagIndices[0]), // int *diagIndices,
        this->devMRGStates, // curandStateMRG32k3a *crStates,
        thrust::raw_pointer_cast(&this->d_hsr_stdDev[0]), // typeReal *siteStddevs,
        thrust::raw_pointer_cast(&this->d_diagInitVals[0]), // typeMat *diagInitVals,
        this->N, //To know when you're past array's end
        this->numCurandStates // int curandLength
    );
    
        
    if(cudaGetLastError()!=cudaSuccess) {
        cout << "ERROR found after addDiagToRealGaussNoise() in updateHamHakenStrobl(). ";
        cout << "Error string: " << cudaGetErrorString(cudaGetLastError()) << endl;
        return false;
    }
    return true;

}



//Update Hamiltonian for Kub-Anderson
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
updateHamKuboAnderson() {

    int threadsPerBlock = 128;
    int numBlocks = (int) ceil( (double)this->N / (double)threadsPerBlock );
    cudaError_t cuErr;

    //Run GPU kernel
    addAndUpdateKANoise<<<numBlocks,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&this->d_hamCsr.csrValA[0]), //typeMat *HamVals,
        thrust::raw_pointer_cast(&this->d_diagIndices[0]), //int *diagIndices,
        thrust::raw_pointer_cast(&this->d_ka_dEps[0]), //typeReal *dEps,
        this->devMRGStates, //curandStateMRG32k3a *crStates,
        thrust::raw_pointer_cast(&this->d_ka_invRelaxTimes[0]), //typeReal *invRelaxTimes,
        thrust::raw_pointer_cast(&this->d_ka_stochCoeffs[0]), //typeReal *stochCoeffs,
        thrust::raw_pointer_cast(&this->d_diagInitVals[0]), //typeMat *diagInitVals,
        this->ka_subSteps, //int numSubSteps,
        this->ka_subdt, //typeReal subdt,
        this->ka_sqrtSubdt, //typeReal sqrt_subdt,
        this->N,  //int vectorSize, //To know when you're past array's end
        this->numCurandStates //int curandLength
    );
    cuErr = cudaGetLastError();
    if(cuErr!=cudaSuccess) {
        cout << "ERROR found after addDiagToRealGaussNoise() in updateHamKuboAnderson(). ";
        cout << "Error string: " << cudaGetErrorString(cuErr) << endl;
        return false;
    }

    return true;
}


//Update Hamiltonian for ZZ Real
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
updateHamZZReal(tReal timeVal) {


    int threadsPerBlock = 128;
    int numBlocks = (int) ceil( (double)this->N / (double)threadsPerBlock );
    cudaError_t cuErr;

    //Run GPU kernel
    updateZZRealNoise<<<numBlocks,threadsPerBlock>>>(
        thrust::raw_pointer_cast(&this->d_hamCsr.csrValA[0]), //typeMat *HamVals,
        thrust::raw_pointer_cast(&this->d_diagIndices[0]), //int *diagIndices,
        thrust::raw_pointer_cast(&this->d_diagInitVals[0]), //typeMat *diagInitVals,
        thrust::raw_pointer_cast(&this->d_phi_n[0]), // typeReal *phiVals,
        thrust::raw_pointer_cast(&this->d_zzOmegaVals[0]), // typeReal *omegaVals,
        thrust::raw_pointer_cast(&this->d_zzFlucCoeffs[0]), // typeReal *flucCoeffs,
        timeVal, // typeReal timeVal,
        this->N,  //int numSites, //To know when you're past array's end
        this->numZZoscPerSite //int totNumOsc
    );
    cuErr = cudaGetLastError();
    if(cuErr!=cudaSuccess) {
        cout << "ERROR found after updateZZRealNoise() in updateHamZZReal(). ";
        cout << "Error string: " << cudaGetErrorString(cuErr) << endl;
        return false;
    }

    //Test by printing out a single site
    // int diagInd = 100;
    // int csrIndex = this->h_diagIndices[diagInd];
    // tMat* ptr = &(this->d_hamCsr.csrValA[csrIndex]);
    // tMat val;
    // cudaMemcpy(&val, ptr, sizeof(tMat), cudaMemcpyDeviceToHost);
    // cout << val.x << " " << val.y << endl;

    return true;

}


//Get total number of loops that you will do
template <typename tMat, typename tState, typename tReal>
int systemClass<tMat,tState,tReal>::
getNumTotalLoops() {

    if(this->doStateLooping) {
        return (int)ceil( (float)(this->totalSteps)/(float)(this->stepsPerLoop) );
    } else {
        return 1;
    }

}




template <typename tMat, typename tState, typename tReal>
int systemClass<tMat,tState,tReal>::
getNumStepsThisLoop(int currentStep) {

    if(this->doStateLooping) {

        if( (currentStep+this->stepsPerLoop) > this->totalSteps ) {
            return this->totalSteps - currentStep;
        } else {
            return this->stepsPerLoop;
        }

    } else {

        return this->totalSteps;

    }


}


//Calculate error estimate and add to total accumulated error
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
calculateErrorEstimate(
    lanczosClass<tMat,tState,tReal> *lanObjPtr,
    padeClass<tMat,tState,tReal> *padeObjPtr,
    taylorKrylovClass<tMat,tState,tReal> *tkObjPtr,
    tReal vecNorm
) {

    //This function uses Algorithm 3.2 from the expokit paper

    tReal errEstimate;

    //Get error from Pade object
    if(this->propMode==PROP_MODE_LANCZOS) {
        errEstimate = padeObjPtr->getKrylovError();
    } else if (this->propMode==PROP_MODE_TAYLOR_LANCZOS) {
        errEstimate = tkObjPtr->getTaylorKrylovError(vecNorm);
    }

    if( errEstimate < DOUBLE_MACHINE_ROUNDOFF ) {
        errEstimate = DOUBLE_MACHINE_ROUNDOFF;
    }

    //Increment error
    this->totalAccumulatedError += errEstimate;

    return true;


}


//Print total error
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::
printTotalError() {

    cout << "Total Accumulated Error for this run: " << this->totalAccumulatedError << "." << endl;

}



//Print out all the state vectors
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
printAllStates() {

    int col, row;

    // cout << "matOfStates.numRows = " << matOfStates.numRows << endl;
    // cout << "matOfStates.numCols = " << matOfStates.numCols << endl;


    for(col=0; col<this->matOfStates.numCols; col++) {
        cout << "State " << col << endl;
        for(row=0; row<this->matOfStates.numRows; row++) {
            cout << this->matOfStates.getHostElement(row,col).x;
            cout << "+ ";
            cout << this->matOfStates.getHostElement(row,col).y;
            cout << "i";
            cout << endl;
        }
        cout << endl;
    }


    return true;

}


//For all state vectors, print out vector*conjugate
template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
printAllStatesNormSq() {

    int col, row;

    // cout << "matOfStates.numRows = " << matOfStates.numRows << endl;
    // cout << "matOfStates.numCols = " << matOfStates.numCols << endl;

    double real, imag;
    for(col=0; col<this->matOfStates.numCols; col++) {
        cout << "|State|^2 " << col << endl;
        for(row=0; row<this->matOfStates.numRows; row++) {
            real = this->matOfStates.getHostElement(row,col).x;
            imag = this->matOfStates.getHostElement(row,col).y;
            cout << real*real + imag*imag;
            //if(row==300) cout << " <-- 300";
            //if(row==301) cout << " <-- 301";
            //if(row==359) cout << " <-- 359";
            cout << endl;
        }
        cout << endl;
    }

    return true;

}



//Print one state all steps
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::
printOnePopulationAllSteps(int popNumber) {

    cout << "Population of element " << popNumber << ", all steps:" << endl;
    int col, row;
    row = popNumber;
    double real, imag;
    for(col=0; col<this->matOfStates.numCols; col++) {
        real = this->matOfStates.getHostElement(row,col).x;
        imag = this->matOfStates.getHostElement(row,col).y;
        cout << real*real + imag*imag;
        //if(row==300) cout << " <-- 300";
        //if(row==301) cout << " <-- 301";
        //if(row==359) cout << " <-- 359";
        cout << endl;
    }
    cout << endl;


}



//Print out one state vector
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::
printSingleState(int state) {

    int row;

    cout << endl << "Single State, state " << state << endl;

    for(row=0; row<this->matOfStates.numRows; row++) {
        cout << this->matOfStates.getHostElement(row,state).x;
        cout << "+ ";
        cout << this->matOfStates.getHostElement(row,state).y;
        cout << "i";
        cout << endl;
    }
    cout << endl;

}




// pow4<T> computes the 4th power of a complex number f(x) -> x^4
template <typename T_ret, typename T_param>
struct pow4
{
    __host__ __device__
        T_ret operator()(const T_param& z) const {
            return pow( z.x*z.x + z.y*z.y ,2);
        }
};


template <typename tMat, typename tState, typename tReal>
bool systemClass<tMat,tState,tReal>::
calcInvPatricRatio(int stateNum, int currentStep) {

    // Run only if option is not set to true
    if( ! this->printPartic ) {
        return true;
    }

    // Setup arguments
    pow4<tReal,tState> unary_op;
    thrust::plus<tReal> binary_op;
    tReal init = 0;

    // This is inefficient, but the state vectors are just arrays, not thrust vectors
    // Copy state to new vector
    tState * statePtr = this->matOfStates.getDeviceColumnPtr(stateNum);
    // thrust::device_vector<tState> d_temp(
    //     thrust::device_ptr<tState> dev_ptr( statePtr ) ,
    //     thrust::device_ptr<tState> dev_ptr( statePtr ) + this->N
    //     );

    // Wrap raw ptr with device_ptr
    thrust::device_ptr<tState> dev_ptr( statePtr );

    // Get participation ratio (not done yet)
    float particRat = thrust::transform_reduce(
        dev_ptr ,
        dev_ptr + this->N,
        unary_op,
        init,
        binary_op
        );

    // Store its inverse ( 1/sum(c_i^4) ) [addition, for whole ensemble]
    this->h_invParticRatio[currentStep] += 1./particRat;

    return true;

}





//Cleanup
template <typename tMat, typename tState, typename tReal>
void systemClass<tMat,tState,tReal>::
cleanup() {

}






































