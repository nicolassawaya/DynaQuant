/*
systemClass.h

Nicolas Sawaya
July 2013
*/




template <typename tMat, typename tState, typename tReal>
class systemClass {

    public:
        //systemClass();
        systemClass();
        // void setCusparseHandle(cusparseHandle_t &csHandle);
        // void setCublasHandle(cublasHandle_t &cbHandle);
        void setCusparseHandle(cusparseHandle_t *csHandle);
        void setCublasHandle(cublasHandle_t *cbHandle);



        bool parseInFile(char* filename);
        bool checkAllInParams();

        bool allocateState();
        bool setInitState();
        bool initStateAsBasisVector(int basisElement);
        bool initStateFromFile(string& stateFileName);
        bool initStateAsUnitSpreadVector();
        //bool resetDeviceToInitialState();

        int getNumInitConditions();
        bool prepNextInitState(int iterInitState);

        bool prepAnalysesForNextRun();
        bool extraAnalysisAfterSingleLoop(int loopNum, int currentStep, int numStepsThisLoop);
        bool extraAnalysisAfterSingleRun(int run);
        bool extraAnalysisAfterEnsemble(int initialConditionNumber);
        bool extraAnalysisEnd();
        bool renormalizeAndStoreNorm(int step);

        bool copyAllStatesBackToHost();
        bool printAllStates();
        bool printAllStatesNormSq();
        void printSingleState(int state);
        void printOnePopulationAllSteps(int popNumber);

        // bool repopulateRealNoise();



        tState* getStateDevPtr(int stateNum);


        //Updating Hamiltonian for different simulation types
        bool updateHamiltonian(tReal timeVal);
        bool copyCsrHamDeviceToHost();


        void cleanup();


        //Library objects
        cusparseHandle_t csHandle;
        cublasHandle_t cbHandle;
        curandGenerator_t crGen;
        cufftHandle cufftPlan;
        // int deviceBeingUsed;

        //http://www.stevenmarkford.com/cuda-get-amount-of-free-global-device-memory/
        //for getting how much free and total device memory

        //System size  //Size of state vector is N, size of matrix is NxN
        int N;
        int initialState;
        bool boolInitFromFile;
        string initStateFileName;

        //Debug flags
        bool debugDiags; //Prints out diagonals from Hamiltonian, to file "diags.debug.out"
        ofstream fDiagDebugs;


        //Limited space for propagation
        bool doStateLooping;
        int stepsPerLoop;
        int totalLoops;
        int stepsInLastLoop;
        bool writingOutPops;
        bool printAllPops;
        bool writeBinaryPops;
        int getNumTotalLoops();
        int getNumStepsThisLoop(int currentStep);
        int popsToPrint[10];

        // Printing (inverse) participation ratio
        bool printPartic;
        // One for each timestep. Averaged over all runs.
        // thrust::device_vector<double> d_invParticRatio;
        thrust::host_vector<double> h_invParticRatio;
        bool calcInvPatricRatio(int stateNum, int currentStep);
        


        //Calculating error from lanczos and pade objects
        tReal totalAccumulatedError;
        bool calculateErrorEstimate(
            lanczosClass<tMat,tState,tReal> *lanObjPtr,
            padeClass<tMat,tState,tReal> *padeObjPtr,
            taylorKrylovClass<tMat,tState,tReal> *tkObjPtr,
            tReal vecNorm
        );
        void printTotalError();


        //For calculating spectra
        spectraClass<tMat,tState,tReal> *specObjPtr; //Put this in sysObj constructor
        string filenameDipoleData;
        string filenamePosData;
        bool doingAbsorbSpec;
        bool doingCDSpec;
        bool currentlyDoingAbsorb;
        bool currentlyDoingCD;


        //curand states
        curandStateMRG32k3a *devMRGStates; //sizeof(curandStateMRG32k3a)=72
        int blockSizeCurand;
        int numBlocksCurand;
        int numCurandStates;
        size_t sizeCurandStates;

        //Type of simulations (const, HSR, KA, )
        simulationType simType; //ENUM
        //Propagation mode (Lanczos, Taylor-Lanczos)
        propagationMode propMode;

        //Taylor-Krylov parameters
        int pTaylorKrylov;      //Order of the Taylor-Krylov calculation
        int mMatBTaylorKrylov;  //m for the auxiliary matrix krylov-part


        //Static Disorder
        bool includeStaticDisorder;
        tReal sigmaStaticDisorder;
        thrust::host_vector<tMat> h_origDiagValsFromFile; //The values read in from file
        bool setupStaticDisorderMemory();
        bool resetStaticDisorder();


        //For non-constant Hamiltonians
        bool prepForNonConstHam();
        bool prepForCurand();
        thrust::host_vector<int> h_diagIndices;
        thrust::host_vector<tMat> h_diagInitVals;
        //on device
        thrust::device_vector<int> d_diagIndices;
        thrust::device_vector<tMat> d_diagInitVals;


        //Temperature, needed for some simulations
        tReal temperature;
        //Actually, temperature not needed on the device at this point
        //__constant__ tReal const_temperature[1];


        //HSR
        bool prepForHsr();
        thrust::host_vector<tReal> h_hsr_stdDev;
        thrust::device_vector<tReal> d_hsr_stdDev;
        string hsr_filenameStdDev;


        //Kubo-Anderson
        bool prepForKA();
        //File for stochastic parameters: relaxation times & sqrt(variances)
        string ka_filenameStochParams;
        int ka_subSteps; //number of substeps to in langevin propagation
        tReal ka_subdt; // dt/ka_subSteps
        tReal ka_sqrtSubdt; // sqrt(dt/ka_subSteps)
        thrust::host_vector<tReal> h_ka_dEps;
        thrust::host_vector<tReal> h_ka_invRelaxTimes;
        thrust::host_vector<tReal> h_ka_stochCoeffs;
        thrust::device_vector<tReal> d_ka_dEps;
        thrust::device_vector<tReal> d_ka_invRelaxTimes;
        thrust::device_vector<tReal> d_ka_stochCoeffs;


        //ZZ Real
        bool prepForZZReal();
        bool zzRealResetRandomPhases();
        //probably only need device-side variables
        string zz_filenameSpecDens;
        int numZZoscPerSite; //number of oscillators
        int numZZoscAllSites; //sum of all oscillators on all sites
        tReal dOmega;
        //what data structure should I use, and which ordering in memory?
        //DO: all phi_n of a single site, then all of the next site, etc
        thrust::device_vector<tReal> d_phi_n; //{site0_phi0,site1_phi0,...,site0_phiN,site1_phiN,...}
        thrust::device_vector<tReal> d_zzOmegaVals; //{site0,site1,...,site0,site1,...}
        thrust::device_vector<tReal> d_zzFlucCoeffs;


        //Object for writing out populations
        //Should be turned on anytime pops are printed, even with ENS_AVG_MODE_NONE
        popPrintClass<tMat,tState,tReal> *popPrintObjPtr;

        // Object for printing out states
        bool writingOutStates;
        int writeStateFreq;
        statePrintClass<tMat,tState,tReal> *statePrintObjPtr;

        //Matrix objects
        matCooClass<tMat> h_hamCoo;
        matCooClass<tMat> d_hamCoo; //Do not delete after conversion
        matCsrClass<tMat> h_hamCsr;
        matCsrClass<tMat> d_hamCsr;

        //State vectors
        denseMatrixClass<tMat> matOfStates;
        tReal *d_stateNorm, *d_stateInvNorm;
        tReal h_norm;

        //Ensemble calculations
        ensembleAvgMode ensembleAvgMode;
        int numEnsembleRuns;

        size_t stateMemSize;
        bool stateAllocated;

        //tMat
        tReal dt;
        tReal* d_dt;
        int totalSteps;
        int m; //Size of decomposition


        //File names
        string hamFileName;
        string tkMatFileName; //The auxiliary matrix for taylor-krylov
        string outFileName;
        string stateFileName;

        //Folder to output everything to
        string outFolderName;
        string popFileName, tempPopFileName;
        fstream popFilePtr, tempPopFilePtr;





    protected:
        bool readInHam();
        //void split(string& str, string& str1, string& str2)

        bool updateHamHakenStrobl();
        bool updateHamKuboAnderson();
        bool updateHamZZReal(tReal timeVal);


};


















