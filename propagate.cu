/*
propagate.cu
Main file for matrix exponential propagation

Nicolas Sawaya
2013 
*/


#include <includeFiles.cu>


int main( int argc, char *argv[] ) {
    
    
    
    //For error handling
    //bool errorThrown = false;

	

    //Read in which device being used
    if(argc<3) {
        cout << "ERROR. Must include device being used in command line. Usage:";
        cout << endl << "./executable {devicenumber}.";
        exit(1);
    }
    int deviceUsed = atoi(argv[2]);
    cout << "Setting device to " << deviceUsed << "." << endl << endl;
    if(cudaSetDevice(deviceUsed)!=cudaSuccess) {
        cout << "ERROR setting device to " << deviceUsed;
        cout << "." << endl;
        return false;
    }




    //Set up CULA, cuBLAS, cuSPARSE
    if( ! initCula() )  exit(1);
    cusparseHandle_t cusparseHandle=0;
    if( ! initCusparse(&cusparseHandle) ) exit(1);
    cublasHandle_t cublasHandle=0;
    if( ! initCublas(&cublasHandle) ) exit(1);
    //cuSPARSE
    //cusparseStatus_t csStatus;
    //cusparseHandle_t cusparseHandle = 0;
    //cusparseMatDescr_t matDescr_A = 0;

	cout << endl;
	cout << " (                                                         \n"
	" )\\ )                        (                          )  \n"
	"(()/(   (               )  ( )\\     (      )         ( /(  \n"
	" /(_))  )\\ )   (     ( /(  )((_)   ))\\  ( /(   (     )\\()) \n"
	"(_))_  (()/(   )\\ )  )(_))((_)_   /((_) )(_))  )\\ ) (_))/  \n"
	" |   \\  )(_)) _(_/( ((_)_  / _ \\ (_))( ((_)_  _(_/( | |_   \n"
	" | |) || || || ' \\))/ _` || (_) || || |/ _` || ' \\))|  _|  \n"
	" |___/  \\_, ||_||_| \\__,_| \\__\\_\\ \\_,_|\\__,_||_||_|  \\__|  \n"
	"        |__/                                               \n";
	
	cout << endl << endl;
	cout << "DynaQuant (beta)" << endl;
	cout << endl;
	cout << "If you use the code, please consider citing Sawaya, et al., Nano Lett., 2015, 15 (3), 1722";
	cout << endl << endl;
	
    
    //Set up timer
    time_t timeStart, timeNow;
    time(&timeStart);
	
	
    //Declare
    systemClass<typeMat,typeState,typeReal> sysObj;//(/*SIM_TYPE_CONST_HAM*/);
    sysObj.setCusparseHandle(&cusparseHandle);
    sysObj.setCublasHandle(&cublasHandle);
    initCula(); //Initialize CULA
	
    
    //Parse in-file
    if( ! sysObj.parseInFile(argv[1]) ) exit(1);
    
    
    //Initialize state
    if( ! sysObj.setInitState() ) exit(1);
    // if( ! sysObj.initStateAsBasisVector(sysObj.initialState) ) exit(1);


    //Get exponential coefficient. This will include dt, equaling dt*i*(-1/hbar)
    typeMat h_expCoeff;
    typeMat *d_expCoeff;
    h_expCoeff = get_neg_inv_hbar_imag();
    h_expCoeff.y = h_expCoeff.y * sysObj.dt;
    cout << "h_expCoeff.{x,y} = " << h_expCoeff.x << ", " << h_expCoeff.y << endl;
    cudaError_t cuErr;
    cuErr = cudaMalloc((void**)&d_expCoeff, sizeof(typeMat));
    cuErr = cudaMemcpy( d_expCoeff, &h_expCoeff, sizeof(typeMat), cudaMemcpyHostToDevice );
    if(cuErr!=cudaSuccess) {
        cout << "ERROR copying (or maybe allocating) expCoeff to device." << endl;
        exit(1);
        return false;
    }


    //Print this first state
    // sysObj.printSingleState(0);

    //Declare pointers. They have to be pointers if I'm not sure that they will become objects.
    lanczosClass<typeMat,typeState,typeReal> *lanczosObjPtr;
    padeClass<typeMat,typeState,typeReal> *padeObjPtr;
    taylorKrylovClass<typeMat,typeState,typeReal> *tkObjPtr;

    //Declared objects depend on propagation mode
    if(sysObj.propMode==PROP_MODE_LANCZOS) {

        cout << "propMode " << "PROP_MODE_LANCZOS" << endl;

        //Initialize lanczos object
        lanczosObjPtr = new lanczosClass<typeMat,typeState,typeReal>( &sysObj,
                     sysObj.m, &(sysObj.d_hamCsr) );

        //Initialize pade object
        padeObjPtr = new padeClass<typeMat,typeState,typeReal>( &sysObj , lanczosObjPtr , d_expCoeff );


    } else if(sysObj.propMode==PROP_MODE_TAYLOR_LANCZOS) {

        cout << "propMode " << "PROP_MODE_TAYLOR_LANCZOS" << endl;

        //Initialize Taylor-Krylov object
        tkObjPtr = new
        taylorKrylovClass<typeMat,typeState,typeReal>(
            &sysObj,
            &(sysObj.d_hamCsr),
            sysObj.tkMatFileName,
            sysObj.pTaylorKrylov,
            sysObj.mMatBTaylorKrylov,
            h_expCoeff,
            d_expCoeff
        );

    }


    //Get number of ensembles to run
    int numEnsembleRuns=1;
    if(sysObj.ensembleAvgMode != ENS_AVG_MODE_NONE ) {
        numEnsembleRuns = sysObj.numEnsembleRuns;
    }
    cout << "numEnsembleRuns = " << numEnsembleRuns << "." << endl << endl;



    //Get number of initial conditions to run (e.g. for abs spec, it's 3)
    int numInitConditions;
    numInitConditions = sysObj.getNumInitConditions();
    cout << "numInitConditions = " << numInitConditions << endl;

    //Show time
    time(&timeNow);
    cout << "Starting simulation. Elapsed time is " << (timeNow-timeStart) << endl;

    //Get number of inner loops you will do
    int totalLoops = sysObj.totalLoops;

    //Loop over different initial conditions.
    for(int iterInitCond=0;iterInitCond<numInitConditions;iterInitCond++){

        //Set next initial state
        cout << "Setting initial state #" << iterInitCond << endl;
        sysObj.prepNextInitState(iterInitCond);


        //Loop over different ensemble runs
        for(int run=0; run<numEnsembleRuns; run++) {

            //cout << "Starting run=" << run << endl;

            //cout << "Setting initial vector." << endl;
            //Reset the inital basis vector
            //if( ! sysObj.resetDeviceToInitialState() ) exit(1);


            //Prepping for next run (including re-randomizing ZZReal Hamiltonian)
            if( !sysObj.prepAnalysesForNextRun() ) return false;

            //Do several "loops" (this saves memory)
            int currentStep = 0;
            for(int loopNum=0; loopNum<totalLoops; loopNum++) {

                //cout << "Doing getNumStepsThisLoop()" << endl;
                //Get number of steps in this loop. Prevents against overshooting end of array.
                int numStepsThisLoop = sysObj.getNumStepsThisLoop(currentStep);


                //cout << "Starting loop" << endl;
                //Complete one loop
                int destVecNum, sourceVecNum;
                for(int step=0; step<numStepsThisLoop; step++) {

                    //Get dest and source vectors, i.e. dest=exp(-iH*dt)*source
                    //Depends on whether this is the first loop or not
                    sourceVecNum = (step==0&&loopNum!=0) ? (sysObj.stepsPerLoop) : (step);
                    destVecNum = step + 1;


                    //Calculate and store inv. partic. ratio, if specified
                    if(! sysObj.calcInvPatricRatio(sourceVecNum, currentStep) ) break;

                    //Update the Hamiltonian (with e.g. stochastic process)
                    if(! sysObj.updateHamiltonian( currentStep*sysObj.dt ) ) return false;

                    //Debug flag for Hamiltonian
                    if(sysObj.debugDiags) {
                        //Copy Hamiltonian back
                        sysObj.h_hamCoo.copyFromDevice(sysObj.d_hamCoo);
                        // sysObj.fDiagDebugs << "** step " << step << " **" << endl;
                        for(int elem=0;elem<sysObj.h_hamCoo.nnz;elem++) {
                            // sysObj.fDiagDebugs << sysObj.h_hamCoo.cooRowIndA[elem] << "  " << sysObj.h_hamCoo.cooColIndA[elem] << endl;
                            if(sysObj.h_hamCoo.cooRowIndA[elem]==sysObj.h_hamCoo.cooColIndA[elem])
                                sysObj.fDiagDebugs << sysObj.h_hamCoo.cooValA[elem].x << endl;
                        }
                    }

                    //Branch based on propagation mode
                    if(sysObj.propMode==PROP_MODE_LANCZOS) {
                        
                        //Do the lanczos decomposition
                        if(! lanczosObjPtr->doLanczos( sysObj.getStateDevPtr(sourceVecNum) ) ) return false;
                        
                        //Do the pade exponentiation
                        if(! padeObjPtr->doPade() ) break;
                        
                        //Propagate exp(-i/hbar * tau * H)*vec = beta*Q*exp(-i/hbar *tau*H)*e1;
                        if(! propagateSystem(&sysObj, lanczosObjPtr, padeObjPtr, destVecNum) ) break;
                        
                        
                    } else if(sysObj.propMode==PROP_MODE_TAYLOR_LANCZOS) {

                        //Propagate using Taylor-Krylov method
                        if(! tkObjPtr->propagateWithTaylorKrylov( 
                            sysObj.getStateDevPtr(sourceVecNum), sysObj.getStateDevPtr(destVecNum)
                            ) ) return false;

                    }

                    //Renormalize (partic ratio calculated here)
                    if(! sysObj.renormalizeAndStoreNorm(destVecNum) ) break;


                    //Calculate Error
                    if(!
                        sysObj.calculateErrorEstimate(lanczosObjPtr,padeObjPtr,tkObjPtr,sysObj.h_norm)
                    ) return false;


                    //Increment
                    currentStep++;

                } //end of one "loop"

                cout << "currentStep = " << currentStep << endl;
                //Run extra routines as necessary
                cout << "run = " << run << endl;
                if(! sysObj.extraAnalysisAfterSingleLoop(loopNum,currentStep,numStepsThisLoop) ) exit(1);


            } //end of single ensemble run


            //Run extra routines as necessary, e.g. routines required for spectra
            if(! sysObj.extraAnalysisAfterSingleRun(run) ) exit(1);

            //Print out total accumulated error for the run
            sysObj.printTotalError();


            //Show time
            time(&timeNow);
            //cout << "Finished run " << run+1 << ". Elapsed time is ";
            cout << (timeNow-timeStart) << " seconds elapsed." << endl;


            //Computes populations, adds to average
            // if(! sysObj.addPopToEnsembleAverage( run, lanczosObj.d_one ) ) break;


        } //end of loop over ensemble runs


        //Average out ensemble if in ensemble-mode
        // sysObj.calcEnsembleAverage();

        //Run extra routines as necessary, e.g. routines required for spectra
        if(! sysObj.extraAnalysisAfterEnsemble(iterInitCond) ) exit(1);


    } //end of loop over different initial conditions


    //Run extra routines as necessary, e.g. routines required for spectra
    if(! sysObj.extraAnalysisEnd() ) exit(1);



    //Copy that state back to host
    //Maybe don't always include
    //sysobj.matOfStates.
    //Copy all states back to host
    // sysObj.matOfStates.copyThisDeviceToThisHost();

    //Print out the states
    //////sysObj.matOfStates.printHostMat();
    // sysObj.printAllStates();
    // sysObj.printAllStatesNormSq();

    //Copy all populations averages back to host
    // sysObj.matEnsembleAverage.copyThisDeviceToThisHost();
    // sysObj.printOneSiteEnsembleAvgPopAllSteps(300);
    //sysObj.printOneSiteEnsembleAvgPopAllSteps(301);
    //sysObj.printOneSiteEnsembleAvgPopAllSteps(359);

    
    

    //Show time
    time(&timeNow);
    cout << endl << "Closing. Elapsed time is " << (timeNow-timeStart);
    cout << " seconds." << endl;
    
}































