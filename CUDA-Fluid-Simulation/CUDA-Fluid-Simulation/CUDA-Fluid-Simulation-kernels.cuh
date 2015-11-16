/*
 * 
 *
 */
#ifndef __STABLEFLUIDS_KERNELS_CUH_
#define __STABLEFLUIDS_KERNELS_CUH_

#include "defines.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include <curand.h>
//#include <curand_kernel.h>

__global__
void advectVelocity_kernel();

#endif
