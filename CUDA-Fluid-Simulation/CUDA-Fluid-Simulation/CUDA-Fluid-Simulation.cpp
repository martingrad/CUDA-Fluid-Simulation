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

StopWatchInterface *timer = NULL;
static int fpsCount = 0;
int fpsLimit = 1;

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
void display(void)
{
	sdkStartTimer(&timer);

	// Render stuff
	glClearColor(0.0, 0.0, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisable(GL_TEXTURE_2D);

	// Finish timing before swap buffers to avoid refresh sync
	sdkStopTimer(&timer);
	glutSwapBuffers();

	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", WINDOW_HEIGHT, WINDOW_HEIGHT, ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}

	glutPostRedisplay();
}

/*
* InitGL
*/
int initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("Compute Stable Fluids");
	glutDisplayFunc(display);
	
	glewInit();

	if (!glewIsSupported(
		"GL_ARB_vertex_buffer_object"
		))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	return true;
}

int main(int argc, char **argv)
{
	void *fluidVelocityData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidVelocityType));
	void *fluidPressureData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidPressureType));
	initCuda(fluidVelocityData, fluidPressureData, volumeSize);
	
	initGL(&argc, argv);
	simulateFluid();

	while (true)
	{
		display();
	}
	
	return 0;
}
