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

float3* cellVelocities;
float* cellPressures;

extern "C" void advectVelocity();

void initFluid()
{
	const int NUMBER_OF_GRID_CELLS = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z;
	cellVelocities = new float3[NUMBER_OF_GRID_CELLS];
	cellPressures = new float[NUMBER_OF_GRID_CELLS];
	
	for(int i = 0; i < NUMBER_OF_GRID_CELLS; ++i)
	{
		cellVelocities[i] = make_float3(0.0, 0.0, 0.0);
		cellPressures[i] = 0.0;
	}

}

/*
* simulateFluid
*/
void simulateFluid()
{
	initFluid();
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
	simulateFluid();
	return 0;
}
