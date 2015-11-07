/*
 * CUDA Fluid Simulation
 * TSBK03 - Teknik för avancerade datorspel
 * Linköping University
 * Martin Gråd and Emma Forsling Parborg
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
#include "shaders.h"

cudaExtent volumeSize = make_cudaExtent(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Y);
const int NUMBER_OF_GRID_CELLS = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z;

StopWatchInterface *timer = NULL;
static int fpsCount = 0;
int fpsLimit = 1;

GLuint shaderProgram = NULL;

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
	// TODO: glEnable() stuff...?

	// TODO: Add quad to draw to and display on screen
	glUseProgram(shaderProgram);
	//glUniform1f(glGetUniformLocation(shaderProgram, "pointScale"), m_window_h / tanf(m_fov*0.5f*(float)M_PI / 180.0f));
	//glUniform1f(glGetUniformLocation(shaderProgram, "pointRadius"), m_particleRadius);
	glUseProgram(0);

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

	// Create shader program
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vertexShaderContents, 0);
	glShaderSource(fragmentShader, 1, &fragmentShaderContents, 0);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success)
	{
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	shaderProgram = program;

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
