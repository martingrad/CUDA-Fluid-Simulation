/*
 *
 *
 */

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA FFT Libraries
#include <cufft.h>

// CUDA helper functions
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include "CUDA-Fluid-Simulation-kernels.cuh"

cudaExtent volumeSize = make_cudaExtent(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Y);
const int NUMBER_OF_GRID_CELLS = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z;

extern "C" void advectVelocity();
extern "C" void initCuda(void *fluidData_velocity, void* fluidData_pressure, cudaExtent volumeSize);

/*
* simulateFluid
*/
void simulateFluid()
{
	//initFluid();
	advectVelocity();
}

/*
* display
*/
void display()
{

}

int main(int argc, char **argv)
{
	void *fluidVelocityData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidVelocityType));
	void *fluidPressureData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidPressureType));
	initCuda(fluidVelocityData, fluidPressureData, volumeSize);
	
	simulateFluid();
	
	return 0;
}
