/*
includeFiles.cu

Nicolas Sawaya
2013
*/


//Definitions
#define DOUBLE_MACHINE_ROUNDOFF 1.110223e-16
#define PI 3.14159265358979323846
#define INVCM_TO_J 1.986446e-23
#define BOLTZ_INVCM 0.695035 // cm^-1/K
//#define HBAR_INVCM 5.3088e-12 // cm^-1 //WRONG. IT'S {ENERGY}/{TIME}
#define FS_TO_CM 1.88365e-4
#define INVFS_TO_INVCM 5308.84

//Enum for simulation types
enum simulationType {
    SIM_TYPE_CONST_HAM,
    SIM_TYPE_HSR,
    SIM_TYPE_KUBO_ANDERSON,
    SIM_TYPE_ZZ_REAL_NOISE, 	//Zhong and Zhao JCP 2011 paper
    SIM_TYPE_ZZ_COMPLEX_NOISE	//Zhong and Zhao JCP 2013 paper
};

//Enum for propagation mode
enum propagationMode {
	PROP_MODE_LANCZOS,
	PROP_MODE_TAYLOR_LANCZOS //Taylor-Krylov with Lanczos algorithm
};

//Enum for ensemble type
enum ensembleAvgMode {
	ENS_AVG_MODE_NONE,
	ENS_AVG_MODE_FIRST,
	ENS_AVG_MODE_CONTINUATION
};

// #define SIM_TYPE_CONST_HAM
// #define 

//Enum for axis (x, y, or z)
// enum axisType {
// 	AXIS_TYPE_X,
// 	AXIS_TYPE_Y,
// 	AXIS_TYPE_Z
// };


#include <iostream>
#include <iomanip>
#include <ctime>
//#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <utility> //things like 'pair' type
//#include <random> //this is supposed to have normal distribution
#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>
//#include <fftw3.h>

using namespace std;

//For creating directory (http://stackoverflow.com/questions/7430248/how-can-i-create-new-folder-using-c-language):
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>


//CUDA libraries
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cufft.h>

//#include <curand.h>
#include <curand_kernel.h>


//CULA
#include <cula_lapack_device.h>

//Type definitions
typedef cuDoubleComplex typeMat;
typedef cuDoubleComplex typeState;
typedef double typeReal;


//Program files


#include <errorHelper.cu>
#include <sparseMatrixClasses.cu>
#include <denseMatrixClasses.cu>
#include <cppUtility.cpp> //My own utilities for non-cuda stuff

#include <gpuSimpleOps.cu>
#include <gpuLanczosHelper.cu>
#include <gpuPadeHelper.cu>
#include <gpuSysHelper.cu>
#include <gpuSpectraHelper.cu>

//#include <cudaHelper.cu>
#include <initLibs.cu>
//Forward declarations:
template <typename tMat, typename tState, typename tReal> class systemClass;
template <typename tMat, typename tState, typename tReal> class lanczosClass;
template <typename tMat, typename tState, typename tReal> class padeClass;
template <typename tMat, typename tState, typename tReal> class taylorKrylovClass;
template <typename tMat, typename tState, typename tReal> class popPrintClass;
template <typename tMat, typename tState, typename tReal> class statePrintClass;
#include <spectraClass.cu>
#include <systemClass.cu>
#include <lanczosClass.cu>
#include <padeClass.cu>
#include <taylorKrylovClass.cu>
#include <popPrintClass.cu>
#include <statePrintClass.cu>

#include <propStep.cu>














