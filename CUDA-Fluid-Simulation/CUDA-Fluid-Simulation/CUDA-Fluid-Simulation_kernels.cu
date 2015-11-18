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

// Global scope surface to bind to
surface<void, cudaSurfaceType3D> surfaceWrite;

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

// Simple kernel to just write something to the texture
__global__
void kernel(dim3 texture_dim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= texture_dim.x || y >= texture_dim.y || z >= texture_dim.z)
	{
		return;
	}
	
	//float4 element = make_float4(0.5f / VOLUME_SIZE_X, 0.0, 1.0, 0.2f);
	//surf3Dwrite(element, surfaceWrite, x * sizeof(float4), y, z);

	/*float4 temp;
	surf3Dread(&temp, surfaceWrite, x * sizeof(float4), y, z);

	float4 tempElement = make_float4(element.x + temp.x, element.w + temp.w, element.z + temp.z, element.w + temp.w);
	surf3Dwrite(temp, surfaceWrite, x * sizeof(float4), y, z);*/
}

/* = External cpp function implementations =
 * =========================================
*/

extern "C"
void advectVelocity()
{
	dim3 threadsPerBlock(pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3), pow(THREADS_PER_BLOCK, 1 / 3));
	dim3 numBlocks(VOLUME_SIZE_X / threadsPerBlock.x, VOLUME_SIZE_Y / threadsPerBlock.y, VOLUME_SIZE_Z / threadsPerBlock.z);
}

extern "C"
void launch_kernel(cudaArray *cuda_image_array, dim3 texture_dim, float testFloatX, float testFloatY, float testFloatZ)
{

	dim3 block_dim(8, 8, 8);
	dim3 grid_dim(texture_dim.x / block_dim.x, texture_dim.y / block_dim.y, texture_dim.z / block_dim.z);

	// Bind voxel array to a writable CUDA surface
	cudaBindSurfaceToArray(surfaceWrite, cuda_image_array);

	// Create the cuda resource description
	struct cudaResourceDesc resoureDescription;
	memset(&resoureDescription, 0, sizeof(resoureDescription));
	resoureDescription.resType = cudaResourceTypeArray;				// be sure to set the resource type to cudaResourceTypeArray
	resoureDescription.res.array.array = cuda_image_array;			// this is the important bit

	// Create the surface object
	cudaSurfaceObject_t writableSurfaceObject = 0;
	cudaCreateSurfaceObject(&writableSurfaceObject, &resoureDescription);

	// Launch kernal operations
	kernel<<<grid_dim, block_dim>>>(texture_dim);
		
	//cutilCheckMsg("kernel failed");
}
