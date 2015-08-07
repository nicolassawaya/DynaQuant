/*
statePrintClass.cu

Handles the printing of raw states.

Nicolas Sawaya
Aspuru-Guzik Group
January 2015
*/










template <typename tMat, typename tState, typename tReal>
class statePrintClass {


public:

	statePrintClass( systemClass<tMat,tState,tReal> *inputSysObjPtr );

	bool printStatesFromLoop(int loopNum, int currentStep, int numStepsThisLoop);
	void closeFiles();

	bool binaryOutput;

	int numStatesToPrint;

	// System object
	systemClass<tMat,tState,tReal> *sysObjPtr;

	// Variable from system
	int writeFreq;
	tReal dt;
	int N;

	// Files and filenames
	vector<float> timestamps;
	vector<string> stateFilenames;
	ofstream *stateFiles;


	// Temporary storage
	thrust::host_vector<tState> tempState;



private:
	statePrintClass();


};


// Convert float to string
string floatToString(float val) {
	stringstream ss;
	ss << val;
	return ss.str();
}




//Default Constructor
template <typename tMat, typename tState, typename tReal>
statePrintClass<tMat,tState,tReal>::
statePrintClass() {
    //Private--it can't be accessed
}


//Constructor
template <typename tMat, typename tState, typename tReal>
statePrintClass<tMat,tState,tReal>::
statePrintClass( systemClass<tMat,tState,tReal> *inputSysObjPtr ) {
    


	// Store system object ptr
	this->sysObjPtr = inputSysObjPtr;

	this->writeFreq = inputSysObjPtr->writeStateFreq;
	this->dt = inputSysObjPtr->dt;
	this->N = inputSysObjPtr->N;

	// Temporary state
	this->tempState.resize(N);

	// Total number of states to print
	// this->numStatesToPrint = 1 + inputSysObjPtr->totalSteps / this->writeFreq;
	this->numStatesToPrint = inputSysObjPtr->totalSteps / this->writeFreq;
	cout << "Number of states to print per ensemble run: " << this->numStatesToPrint;
	cout << endl;
	cout << "writeFreq: " << this->writeFreq << endl;
	cout << "totalSteps: " << inputSysObjPtr->totalSteps << endl << endl;

	this->timestamps.resize(this->numStatesToPrint);
	this->stateFilenames.resize(this->numStatesToPrint);
	this->stateFiles = new ofstream[this->numStatesToPrint];


	// Files based on timesteps and frequency of printing
	for ( int i=0 ; i < this->numStatesToPrint ; i++ ) {

		float stamp = this->dt * i * this->writeFreq;
		this->timestamps[i] = stamp;

		string fname = inputSysObjPtr->outFolderName + "/_states_";
		fname = fname + floatToString(stamp) + ".dat";
		// cout << "fname: " << fname << endl;
		// string fname = "_states_" + floatToString(stamp) + ".dat";
		this->stateFilenames[i] = fname;

		this->stateFiles[i].open(fname.c_str(), ios::out | ios::binary );
        
		// Print header
		// Header: {timestamp}, {numSites}, {numruns}
		this->stateFiles[i].write( (char*) &(stamp) , sizeof(float) );
		this->stateFiles[i].write( (char*) &(this->N) , sizeof(int) );
		this->stateFiles[i].write( (char*) &(inputSysObjPtr->numEnsembleRuns) , sizeof(int) );

		// New *****
		this->stateFiles[i].close();
	}




}


// Print states from loop
template <typename tMat, typename tState, typename tReal>
bool statePrintClass<tMat,tState,tReal>::
printStatesFromLoop(int loopNum, int currentStep, int numStepsThisLoop) {

	// Function assumes correct states already present on host

	// Overall step in simulation
	int overallSimStep = loopNum * this->sysObjPtr->stepsPerLoop;

	cout << "Inside printStatesFromLoop." << endl;

	// Easiest is to just loop through
	for (int substep=0; substep<numStepsThisLoop; substep++) {


		// cout << "numStepsThisLoop = " << endl;

		// Check if state is one to print
		if ( overallSimStep % this->writeFreq == 0 ) {

			int printId = overallSimStep / this->writeFreq;

			tState *statePtr;
			if (loopNum==0) { 

				// Element zero is always the initial state,
				// even for later loops.
				// So after the first loop, we have to add 
				// one to the subsequent ones.
				statePtr = this->sysObjPtr->matOfStates.getHostColumnPtr(substep);
				
			} else {

				statePtr = this->sysObjPtr->matOfStates.getHostColumnPtr(substep + 1);

			}

			// // for debug *****
			// for (int i=0;i<this->N;i++) {
			// 	if (printId==1) cout << statePtr[i].x << "  " << endl;
			// 	// this->stateFiles[printId].write( (char*) &(statePtr[i]) , sizeof(tState) );
			// }

			ofstream f( this->stateFilenames[printId].c_str() , ios::out | ios::binary | ios::app );
			f.write( (char*) &(statePtr[0]) , this->N*sizeof(tState) );
			f.close();


		}

		// Increment step
		overallSimStep++;


	}

	return true;


}

// //Print states from loop
// template <typename tMat, typename tState, typename tReal>
// bool statePrintClass<tMat,tState,tReal>::
// printStatesFromLoop(int loopNum, int currentStep, int numStepsThisLoop) {

// 	// Function assumes correct states already present on host

// 	// Overall step in simulation
// 	int overallSimStep = loopNum * this->sysObjPtr->stepsPerLoop;

// 	cout << "Inside printStatesFromLoop." << endl;

// 	// Easiest is to just loop through
// 	for (int substep=0; substep<numStepsThisLoop; substep++) {


// 		// Check if state is one to print
// 		if ( overallSimStep % this->writeFreq == 0 ) {

// 			int printId = overallSimStep / this->writeFreq;

// 			tState *statePtr = this->sysObjPtr->matOfStates.getHostColumnPtr(substep);

// 			// for (int i=0;i<this->N;i++) {
// 			// 	if (printId==1) cout << statePtr[i].x << "  " << endl;
// 			// 	// this->stateFiles[printId].write( (char*) &(statePtr[i]) , sizeof(tState) );
// 			// }

// 			this->stateFiles[printId].write( (char*) &(statePtr[0]) , this->N*sizeof(tState) );

// 		}

// 		// Increment step
// 		overallSimStep++;


// 	}

// 	return true;


// }



//Print states from loop
template <typename tMat, typename tState, typename tReal>
void statePrintClass<tMat,tState,tReal>::
closeFiles() {

	for ( int fid=0; fid<this->numStatesToPrint; fid++ ) {

		// New *****
		// stateFiles[fid].close();

	}


}































