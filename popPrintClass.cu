/*
popPrintClass.cu

Handles the printing of populations, and appending to already-printed populations.
Note that populations are summed only. The averages are NOT taken.

Nicolas Sawaya
Aspuru-Guzik Group
October 2013
*/





template <typename tMat, typename tState, typename tReal>
class popPrintClass {
    
    public:
        popPrintClass( systemClass<tMat,tState,tReal> *inputSysObjPtr );

        bool prepForNextRun();

        // bool printBatch(int batchSize, tReal* batchPtr, systemClass<tMat,tState,tReal> *inputSysObjPtr);
        bool printOneLoopAllPops(int beginningStepNum, int numSteps, systemClass<tMat,tState,tReal> *sysObj);

        bool printOneLoopSomePops();

        void beginningLineError(string strExpected, string strActual);

        bool doTasksForFinishedRun();

        bool prevSimMatchesCurrentSim;

        bool runIsContinuation; //Set to true if continuing a run. Also set to true after first run.
        bool printAllPops;
        bool writeBinaryPops;

        // int simsToRunThisExec;
        int overallSimCounter;
        int totalSteps;
        int numPrintedSites;
        int N; //From sysObj, just the number of elements in the vector
        int* popsToPrintPtr;
        tReal dt;

        int currentStep;
        int currentPop;

        //Pointer to matrix of states
        tState *statesPtr;

        //This is where you hold the previous populations that you read in periodically
        tReal* prevPops;
        int lenPrevPops;

        int stepsPerLoop;

        //Population names and files
        string outFolderName;
        string popFileName, tempPopFileName;
        ifstream popFile;
        ofstream tempPopFile;

    private:
        popPrintClass(); //This way no one can access this constructor

};




//Default Constructor
template <typename tMat, typename tState, typename tReal>
popPrintClass<tMat,tState,tReal>::
popPrintClass() {
    //Private--it can't be accessed
}


//Constructor
template <typename tMat, typename tState, typename tReal>
popPrintClass<tMat,tState,tReal>::
popPrintClass( /*bool runIsContinuation, string outFolderName,*/ systemClass<tMat,tState,tReal> *inputSysObjPtr ) {
    

    cout << "In constructor for popPrintClass()." << endl << endl;

    //Set params from system object
    this->outFolderName = inputSysObjPtr->outFolderName;


    if(inputSysObjPtr->ensembleAvgMode==ENS_AVG_MODE_NONE || inputSysObjPtr->ensembleAvgMode==ENS_AVG_MODE_FIRST) {
        this->runIsContinuation = false;
        cout << "This is _not_ a continuation run." << endl << endl;
    } else if (inputSysObjPtr->ensembleAvgMode==ENS_AVG_MODE_CONTINUATION) {
        this->runIsContinuation = true;
        cout << "This run is a continuation run." << endl << endl;
    }

    this->outFolderName = inputSysObjPtr->outFolderName;

    this->dt = inputSysObjPtr->dt;
    this->totalSteps = inputSysObjPtr->totalSteps;
    //Just point to the system array
    this->printAllPops = inputSysObjPtr->printAllPops;
    this->popsToPrintPtr = inputSysObjPtr->popsToPrint;
    this->writeBinaryPops = inputSysObjPtr->writeBinaryPops;

    //Still storing the total number of sites
    this->numPrintedSites = inputSysObjPtr->N;
    this->N = inputSysObjPtr->N;
    this->stepsPerLoop = inputSysObjPtr->stepsPerLoop;

    //Pointer to matrix of states
    this->statesPtr = inputSysObjPtr->matOfStates.getHostColumnPtr(0);

    //Default population file name
    this->popFileName = "pops.dat";
    this->tempPopFileName = string("temp_") + this->popFileName;
    this->popFileName = inputSysObjPtr->outFolderName + string("/") + this->popFileName;
    this->tempPopFileName = inputSysObjPtr->outFolderName + string("/") + this->tempPopFileName;
    cout << "popFileName = " << this->popFileName << endl;
    cout << "tempPopFileName = " << this->tempPopFileName << endl << endl;

    //Other defaults
    this->prevSimMatchesCurrentSim = false;

    //Allocate for previous populations (you'll read them in each step)
    this->lenPrevPops = this->numPrintedSites * this->stepsPerLoop;
    // prevPops = new tReal[lenPrevPops];
    cout << "this->lenPrevPops = " << this->lenPrevPops << endl;
    prevPops = (tReal*) malloc(lenPrevPops * sizeof(tReal));

    //Determine number of printed sites, and compare to previous run
    this->numPrintedSites = 0;
    if(this->printAllPops) {
        this->numPrintedSites = inputSysObjPtr->N;
    } else {
        int arrlen = sizeof(inputSysObjPtr->popsToPrint)/sizeof(*(inputSysObjPtr->popsToPrint));
        for(int i=0;i<arrlen;i++) {
            if(inputSysObjPtr->popsToPrint[i] != -1) {
                numPrintedSites++;
            } else {
                break;
            }
        }
    }



}





//Prepare for next ensemble run
template <typename tMat, typename tState, typename tReal>
bool popPrintClass<tMat,tState,tReal>::
prepForNextRun() {



    //Binary or non-binary printing?
    if(this->writeBinaryPops) {

        //Sizes for reading and writing
        size_t sizeInt = sizeof(int);
        size_t sizeReal = sizeof(tReal);

        //Open population file from before
        if(this->runIsContinuation) {

            this->popFile.open(this->popFileName.c_str(), ios::in | ios::binary);
            //Read first bytes, and check to see if initial parameters match
            //How do I throw an error? Just throw an error once the first batch starts to print

            //Parameter order:
            //numsims (int)
            //dt (double)
            //totalsteps (int)
            //numprintedsites (int)
            //double double double double...
            int prevNumSims, prevTotalSteps, prevNumPrintedSites;
            tReal prevDt;


            this->popFile.read( (char*) &prevNumSims, sizeInt);
            this->popFile.read( (char*) &prevDt, sizeReal);
            this->popFile.read( (char*) &prevTotalSteps, sizeInt);
            this->popFile.read( (char*) &prevNumPrintedSites, sizeInt);

            this->overallSimCounter = prevNumSims + 1;

            cout << "From previous file:" << endl;
            cout << "{prevNumSims, dt, totalsteps, numprintedsites}:" << endl;
            cout << "{ " << prevNumSims << ", " << prevDt << ", " << prevTotalSteps << ", ";
            cout << prevNumPrintedSites << " }";
            cout << "Current simulation:" << endl;
            cout << "{ []," << this->dt << ", " << this->totalSteps << ","<< this->numPrintedSites;
            cout << " }" << endl;

            //Set flag
            this->prevSimMatchesCurrentSim = true;

            //Make sure parameters from previous population file are the same
            if( this->dt!=prevDt || this->totalSteps!=prevTotalSteps || 
                this->numPrintedSites!=prevNumPrintedSites ) {

                this->prevSimMatchesCurrentSim = false;
                cout << "ERROR. Previous and current simulation parameters do not match." << endl;
                cout << "{dt, totalsteps, numprintedsites}:" << endl;
                cout << "Read in from previous population file:" << endl;
                cout << "{ " << prevDt << ", " << prevTotalSteps << ", ";
                cout << prevNumPrintedSites << " }";
                cout << "Current simulation:" << endl;
                cout << "{ " << this->dt << ", " << this->totalSteps <<","<< this->numPrintedSites;
                cout << " }" << endl;
                cout << "Aborting." << endl;

                return false;
            }

        } else {
            this->prevSimMatchesCurrentSim = true;
            this->overallSimCounter = 1;
        }

        //Open new temporary population file to write to
        this->tempPopFile.open(this->tempPopFileName.c_str(), ios::out | ios::binary);

        //Print header on temp file
        this->tempPopFile.write( (char*) &(this->overallSimCounter) , sizeInt);
        this->tempPopFile.write( (char*) &(this->dt) , sizeReal);
        this->tempPopFile.write( (char*) &(this->totalSteps) , sizeInt);
        this->tempPopFile.write( (char*) &(this->numPrintedSites) , sizeInt);



    } else { //Non-binary writing (text-readable mode)

        //Open population file from before
        if(this->runIsContinuation) {

            this->popFile.open(this->popFileName.c_str());
            //Read first lines, and check to see if initial parameters match
            //How do I throw an error? Just throw an error once the first batch starts to print

            this->prevSimMatchesCurrentSim = true;
            string line, strTemp;
            tReal prevDt;
            int prevTotalSteps, prevNumPrintedSites, prevNumSims;
            stringstream ss;

            //numsims line
            getline(this->popFile,line);
            ss << line;
            ss >> strTemp;
            if(strTemp!="numsims") {
                this->beginningLineError("numsims",strTemp); return false;
            }
            ss >> prevNumSims;
            this->overallSimCounter = prevNumSims + 1; //Output of first 

            //dt line
            ss.str(""); ss.clear(); //Reset stringstream (I hope this works)
            getline(this->popFile,line);
            ss << line;
            ss >> strTemp;
            if(strTemp!="dt") {
                this->beginningLineError("dt",strTemp); return false;
            }
            ss >> prevDt;

            //totalsteps line
            ss.str(""); ss.clear(); //Reset stringstream
            getline(this->popFile,line);
            ss << line;
            ss >> strTemp;
            if(strTemp!="totalsteps") {
                this->beginningLineError("totalsteps",strTemp); return false;
            }
            ss >> prevTotalSteps;

            //numprintedsites line
            ss.str(""); ss.clear();
            getline(this->popFile,line);
            ss << line;
            ss >> strTemp;
            if(strTemp!="numprintedsites") {
                this->beginningLineError("numprintedsites",strTemp); return false;
            }
            ss >> prevNumPrintedSites;

            //Set flag
            this->prevSimMatchesCurrentSim = true;

            //Make sure parameters from previous population file are the same
            if( this->dt!=prevDt || this->totalSteps!=prevTotalSteps || 
                this->numPrintedSites!=prevNumPrintedSites ) {

                this->prevSimMatchesCurrentSim = false;
                cout << "ERROR. Previous and current simulation parameters do not match." << endl;
                cout << "{dt, totalsteps, numprintedsites}:" << endl;
                cout << "Read in from previous population file:" << endl;
                cout << "{ " << " }";
                cout << "Current simulation:" << endl;
                cout << "{ " << this->dt << ", " << this->totalSteps << this->numPrintedSites;
                cout << " }" << endl;
                cout << "Aborting." << endl;

                return false;
            }

        } else {
            this->prevSimMatchesCurrentSim = true;
            this->overallSimCounter = 1;
        }

        //Open new temporary population file to write to
        this->tempPopFile.open(this->tempPopFileName.c_str());


        //Print header on temp file
        this->tempPopFile << "numsims " << this->overallSimCounter << endl;
        this->tempPopFile << "dt " << this->dt << endl;
        this->tempPopFile << "totalsteps " << this->totalSteps << endl;
        this->tempPopFile << "numprintedsites " << numPrintedSites << endl;

    } //end if this->writeBinaryPops

    return true;

}






//Write out an error
template <typename tMat, typename tState, typename tReal>
void popPrintClass<tMat,tState,tReal>::
beginningLineError(string strExpected, string strActual) {

    cout << "ERROR reading beginning lines in popPrinterClass()." << endl;
    cout << "Expected " << strExpected << ", got " << strActual << "." << endl;

}


//Print a batch of all populations
template <typename tMat, typename tState, typename tReal>
bool popPrintClass<tMat,tState,tReal>::
printOneLoopAllPops(int beginningStepNum, int numSteps, systemClass<tMat,tState,tReal> *sysObj) {

    //Note the numSteps might be less than the loopNum for the last loop

    //If mismatch was discovered in startup, abort here
    if(! this->prevSimMatchesCurrentSim) {
        cout << "Previous simulation params do not match current params. Aborting.";
        return false;
    }

    //Size for writing
    size_t sizeReal = sizeof(tReal);


    //Read in Populations from input file
    if(this->runIsContinuation) {

        string line;
        stringstream ss;

        int index;

        //Loop over the loop-steps
        for(int step=0; step<numSteps; step++) {

            //Binary or non-binary printing?
            if(this->writeBinaryPops) {


                for(int site=0;site<this->numPrintedSites; site++) {

                    index = (this->numPrintedSites * step) + site;

                    if(index >= this->lenPrevPops) {
                        cout << "ERROR in printOneLoopAllPops()." << endl;
                        cout << "index = " << index;
                        cout << ", lenPrevPops = " << this->lenPrevPops << endl;
                        return false;
                    }
                    //Put value in prevPops array
                    this->popFile.read( (char*) &this->prevPops[index], sizeReal );

                }


            } else { //Non-binary file

                //This is the "STEP #" line
                getline(this->popFile,line);
                
                for(int site=0; site<this->numPrintedSites; site++) {

                    getline(this->popFile,line);
                    ss << line;

                    index = (this->numPrintedSites * step) + site;

                    if(index >= this->lenPrevPops) {
                        cout << "ERROR in printOneLoopAllPops()." << endl;
                        cout << "index = " << index;
                        cout << ", lenPrevPops = " << this->lenPrevPops << endl;
                        return false;
                    }
                    //Put value in prevPops array
                    ss >> this->prevPops[index];
                    
                    //Clear stringstream
                    ss.str(""); ss.clear();

                } 


            } //End binary-file conditional

        } //end loop over steps


    }




    //Loop through states
    tMat stateVal;
    tReal popVal;
    int index;
    for(int step=beginningStepNum+1; step < (beginningStepNum+numSteps) + 1; step++) {
        //It's beginningStepNum+1 because the first vector is the initial vector (numCols = e.g. 101)

        if( ! this->writeBinaryPops ) {
            this->tempPopFile << "STEP " << step << endl;
        }

        //Loop through sites
        // cout << "numPrintedSites = " << numPrintedSites << endl;
        for(int site=0; site<this->numPrintedSites; site++) {

            int loopStep = step - beginningStepNum;
            index = (step - beginningStepNum)*N + site;
            // cout << "loopStep " << loopStep << ", site " << site << ", index " << index << endl;
            // popVal = pow(this->statesPtr[index].x,2) + pow(this->statesPtr[index].y,2);
            stateVal = sysObj->matOfStates.getHostElement(site,loopStep);
            popVal = pow(stateVal.x,2) + pow(stateVal.y,2);

            //Append previous population
            if(this->runIsContinuation) popVal = popVal + this->prevPops[index - this->N];

            if(this->writeBinaryPops) {
                this->tempPopFile.write( (char*)&popVal, sizeReal );
            } else {
                this->tempPopFile << setprecision(17) << setw(25);
                this->tempPopFile << popVal << " (" << site << ")" << endl;
            }

        }

    }

    return true;

}

//Print a batch of some populations
template <typename tMat, typename tState, typename tReal>
bool popPrintClass<tMat,tState,tReal>::
printOneLoopSomePops() {

    cout << "Code for writing out only some populations out is not written yet. Aborting." << endl;
    return false;

}


//Certain things are done after 
template <typename tMat, typename tState, typename tReal>
bool popPrintClass<tMat,tState,tReal>::
doTasksForFinishedRun() {

    //Close files
    this->tempPopFile.close();
    if(this->popFile.is_open()) this->popFile.close();

    //Rename and delete files
    if(this->runIsContinuation) {
        if( remove(this->popFileName.c_str()) != 0 ) {
            cout << "ERROR removing " << this->popFileName << ". Aborting." << endl;
            return false;
        }   
    }

    if( rename(this->tempPopFileName.c_str(), this->popFileName.c_str()) !=0 ) {
        cout << "ERROR renaming " << this->tempPopFileName << " to ";
        cout << this->popFileName << ". Aborting." << endl;
        return false;
    }

    //Set this flag to make sure the next run adds the populations from before
    this->runIsContinuation = true;

    // Increment overall simulation number
    // This is actually done automatically in the prepfornext function
    // this->overallSimCounter ++;


    return true;

}




// Format of file:
//
// # numsims {number of sims added together}
// # dt {timestep size}
// # totalsteps {number of timesteps}
// # numprintedsites {number of printed sites per timestep}
// # STEP 0
// # (0) {pop sum}
// # (1) {pop sum}
// # ...
// # STEP 1
// # (0) {pop sum}
// # (1) {pop sum}
// # ...
// # ...



















