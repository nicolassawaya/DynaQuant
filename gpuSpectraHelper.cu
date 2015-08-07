/*
gpuSpectraHelper.cu

GPU functions used in calculating spectra.

Nicolas sawaya
July 2013
*/






//Kernel for lineShapeDressing
__global__ void kernelGaussianLineDress(
    typeState *timeCorrFunc, //Function to dress, complex
    typeReal sig,        
    typeReal dt,        
    int totalElems  		//So you know when to stop
) {

	int block = blockIdx.x;
	int thread = threadIdx.x;
	int ind = block*blockDim.x + thread;

	//Ensure element is within limits
	if(ind < totalElems) {

		double t = dt*(ind+1);
		//sqrt(2*pi) = 2.506628...
		//double gauss = (1./ (sig*2.5066282746310002) ) * exp(-0.5 * (t/sig)*(t/sig));
		double gauss = exp(-0.5 * (t/sig)*(t/sig));
		timeCorrFunc[ind].x = timeCorrFunc[ind].x * gauss;
		timeCorrFunc[ind].y = timeCorrFunc[ind].y * gauss;

	}




}



















