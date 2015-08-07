/*
gpuSysHelper.cu

Nicolas Sawaya
July 2013
*/


//Kernel for setting up pseudo-random number generator
__global__ void setupPRNG(int inputSeed, curandStateMRG32k3a *state) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    // curand_init(0, id, 0, &state[id]);
    //"Sequences generated with the same seed and different sequence 
    //numbers will not have statistically correlated values."
    curand_init(
        inputSeed, //seed
        id, //sequence (there are several 'sequences' running in parallel)
        0, //offset
        &state[id] //curandState_t *state
        );

}




//Kernel for updating HSR Hamiltonian
__global__ void addDiagToRealGaussNoise(
    typeMat *HamVals,
    int *diagIndices,
    // float *gaussNoiseArrSD1,
    curandStateMRG32k3a *crStates,
    typeReal *siteStddevs,
    typeMat *diagInitVals,
    int vectorSize, //To know when you're past array's end
    int curandLength
    ) {


    int elem = blockIdx.x*blockDim.x + threadIdx.x;
    

    if(elem<vectorSize) {

        //Take modulus so you don't go past cuRand memory limit
        int crId = elem % curandLength;

        /* Copy state to local memory for efficiency */
        curandStateMRG32k3a localState = crStates[crId];
        float randNorm = curand_normal(&localState);
    
        //typeReal noise = (typeReal)gaussNoiseArrSD1[elem] * siteStddevs[elem];
        typeReal noise = (typeReal)randNorm * siteStddevs[elem];

        HamVals[ diagIndices[elem] ].x = noise + diagInitVals[elem].x;

        /* Copy state back to global memory */
        crStates[crId] = localState;
    }


}





//Kernel for updating Kubo-Anderson's delta_e and Hamiltonian
__global__ void addAndUpdateKANoise(
    typeMat *HamVals,
    int *diagIndices,
    typeReal *dEps,
    curandStateMRG32k3a *crStates,
    typeReal *invRelaxTimes,
    typeReal *stochCoeffs,
    typeMat *diagInitVals,
    int numSubSteps,
    typeReal subdt, //this is dt/numSubSteps. No division in kernel.
    typeReal sqrt_subdt,
    int vectorSize, //To know when you're past array's end
    int curandLength
    ) {


    int elem = blockIdx.x*blockDim.x + threadIdx.x;

    if(elem<vectorSize) {

        //Take modulus so you don't go past cuRand memory limit
        int crId = elem % curandLength;

        /* Copy state to local memory for efficiency */
        curandStateMRG32k3a localState = crStates[crId];
        float randNorm;
        
        //Get current dEps
        typeReal cur_dEps = dEps[elem];
        //Get parameters
        typeReal invTau = invRelaxTimes[elem];
        typeReal coeff = stochCoeffs[elem];

        for(int i=0;i<numSubSteps;i++) {

            //Get random from Normal(1,0)
            randNorm  = curand_normal(&localState);

            //Propagate Langevin equation using Euler-Maruyama Scheme
            cur_dEps = cur_dEps - cur_dEps*invTau*subdt + coeff*randNorm*sqrt_subdt;

        }

        //Write back dEps
        dEps[elem] = cur_dEps;

        //Update Hamiltonian
        HamVals[ diagIndices[elem] ].x = cur_dEps + diagInitVals[elem].x;

        /* Copy state back to global memory */
        crStates[crId] = localState;

    }


}




//Kernel for updating Hamiltonian for ZZReal
__global__ void updateZZRealNoise(
    typeMat *HamVals,
    int *diagIndices,
    typeMat *diagInitVals,
    typeReal *phiVals,
    typeReal *omegaVals,
    typeReal *flucCoeffs,
    typeReal timeVal,
    int numSites, //To know when you're past array's end
    int totNumOscPerSite //number of oscillators per site
    ) {


    int site = blockIdx.x*blockDim.x + threadIdx.x;
    


    if(site<numSites) {

        
        typeReal omega, coeff, phi;
        typeReal noise = 0;
        int index;

        for(int osc=0; osc<totNumOscPerSite; osc++) {
            index = numSites*osc + site;
            
            // omega = omegaVals[index];
            // coeff = flucCoeffs[index];
            // phi = phiVals[index];

            omega = omegaVals[osc];
            coeff = flucCoeffs[osc];
            phi = phiVals[index];

            noise = noise + coeff*cos(omega*timeVal + phi);
        }

        HamVals[ diagIndices[site] ].x = noise + diagInitVals[site].x;

    }


}





//Kernel for adding up the populations of each ensemble run
__global__ void kernel_addPopToEnsembleAvg(
    bool isFirstInEnsemble,
    typeMat *stateMat,
    typeReal *ensemblePopMat,
    int numElems) {


    int elem = blockIdx.x*blockDim.x + threadIdx.x;

    //Ensure element is within limits
    if(elem < numElems) {
        typeReal thisPop;
        if(isFirstInEnsemble) {
            thisPop = 0.;
        } else {
            thisPop = ensemblePopMat[elem];
        }

        ensemblePopMat[elem] = thisPop
                                + pow(stateMat[elem].x,2)
                                + pow(stateMat[elem].y,2);

    }


}    





//Kernel for dividing ensemble sum by number of runs
__global__ void divideForEnsembleAvg(
    typeReal *ensemblePopMat,
    typeReal inverseNumRuns,
    int numElems
    ) {

    int elem = blockIdx.x*blockDim.x + threadIdx.x;

    //Ensure element is within limits
    if(elem < numElems) {
        ensemblePopMat[elem] = ensemblePopMat[elem] * inverseNumRuns;
    }


}

















