/*
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>		// Helper functions for CUDA Error handling

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

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *fluidData_velocity_GPU = 0;
cudaArray *fluidData_pressure_GPU = 0;

// 3D textures
texture<fluidPressureType, 3, cudaReadModeNormalizedFloat> tex_pressure;
texture<fluidVelocityType, 3, cudaReadModeNormalizedFloat> tex_velocity;

/*
* initCuda
*/
extern "C"
void initCuda(void *fluidData_velocity, void* fluidData_pressure, cudaExtent volumeSize)
{
	// Velocity data
	// create 3D array
	cudaChannelFormatDesc channelDesc_v = cudaCreateChannelDesc<fluidVelocityType>();
	checkCudaErrors(cudaMalloc3DArray(&fluidData_velocity_GPU, &channelDesc_v, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams_v = { 0 };
	copyParams_v.srcPtr = make_cudaPitchedPtr(fluidData_velocity, volumeSize.width*sizeof(fluidVelocityType), volumeSize.width, volumeSize.height);
	copyParams_v.dstArray = fluidData_velocity_GPU;
	copyParams_v.extent = volumeSize;
	copyParams_v.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams_v));

	// set texture parameters
	tex_velocity.normalized = true;							// access with normalized texture coordinates
	tex_velocity.filterMode = cudaFilterModeLinear;			// linear interpolation
	tex_velocity.addressMode[0] = cudaAddressModeClamp;		// clamp texture coordinates
	tex_velocity.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex_velocity, fluidData_velocity_GPU, channelDesc_v));

	// Pressure data
	// create 3D array
	cudaChannelFormatDesc channelDesc_p = cudaCreateChannelDesc<fluidPressureType>();
	checkCudaErrors(cudaMalloc3DArray(&fluidData_pressure_GPU, &channelDesc_p, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams_p = { 0 };
	copyParams_p.srcPtr = make_cudaPitchedPtr(fluidData_pressure, volumeSize.width*sizeof(fluidPressureType), volumeSize.width, volumeSize.height);
	copyParams_p.dstArray = fluidData_pressure_GPU;
	copyParams_p.extent = volumeSize;
	copyParams_p.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams_p));

	// set texture parameters
	tex_pressure.normalized = true;							// access with normalized texture coordinates
	tex_pressure.filterMode = cudaFilterModeLinear;			// linear interpolation
	tex_pressure.addressMode[0] = cudaAddressModeClamp;		// clamp texture coordinates
	tex_pressure.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex_pressure, fluidData_pressure_GPU, channelDesc_p));
}

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
void advectVelocity_kernel(char *a, int *b)
{
	// voxel (i,j,k)
	//int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	//int k = (blockIdx.z * blockDim.z) + threadIdx.z;

	a[threadIdx.x] += b[threadIdx.x];
}


/* = External cpp function implementations 
 * =========================================
*/

extern "C"
void advectVelocity()
{
	dim3 threadsPerBlock(pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3));
	dim3 numBlocks(VOLUME_SIZE_X / threadsPerBlock.x, VOLUME_SIZE_Y / threadsPerBlock.y, VOLUME_SIZE_Z / threadsPerBlock.z);
	//dim3 numBlocks(NUMBER_OF_BLOCKS ^ 1/3 /, NUMBER_OF_BLOCKS ^ 1 / 3, NUMBER_OF_BLOCKS ^ 1 / 3);

	/**
	* Ingemar Ragnemalm's 'CUDA Hello World'
	*/

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

	// Call kernel
	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	advectVelocity_kernel<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
	cudaFree(ad);
	cudaFree(bd);

	printf("%s\n", a);
}
