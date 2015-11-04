/*
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>          // CUDA FFT Libraries
#include <helper_cuda.h>    // Helper functions for CUDA Error handling

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// FluidsGL CUDA kernel definitions
#include "CUDA-Fluid-Simulation-kernels.cuh"

/*
* Forward Euler
* x^n+1 = x^n + f(x^n,t^n)t
* The value of x at the next time step equals the current value of x plus the current rate of change,
* times the duration of the time step t
*/
__device__
void forwardEuler()
{

}


__global__
void advectVelocity_GPU(char *a, int *b)
{
	// voxel (i,j,k)
	//int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int k = (blockIdx.z * blockDim.z) + threadIdx.z;

	a[threadIdx.x] += b[threadIdx.x];
	
}

extern "C"
void advectVelocity()
{
	dim3 threadsPerBlock(pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3));
	dim3 numBlocks(VOLUME_SIZE_X / threadsPerBlock.x, VOLUME_SIZE_Y / threadsPerBlock.y, VOLUME_SIZE_Z / threadsPerBlock.z);
	//dim3 numBlocks(NUMBER_OF_BLOCKS ^ 1/3 /, NUMBER_OF_BLOCKS ^ 1 / 3, NUMBER_OF_BLOCKS ^ 1 / 3);

	//advectVelocity_GPU <<<numBlocks, threadsPerBlock>>>(1,2);

	const int N = 16;
	const int blocksize = 16;
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = { 15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

	cudaMalloc((void**)&ad, csize);
	cudaMalloc((void**)&bd, isize);
	cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	advectVelocity_GPU<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
	cudaFree(ad);
	cudaFree(bd);

	printf("%s\n", a);
}