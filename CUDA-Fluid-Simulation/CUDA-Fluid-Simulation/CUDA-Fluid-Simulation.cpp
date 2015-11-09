/*
 * CUDA Fluid Simulation
 * TSBK03 - Teknik f�r avancerade datorspel
 * Link�ping University
 * Martin Gr�d and Emma Forsling Parborg
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

#include <GL/gl.h>
#include <GL/glu.h>

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

//#define GLEW_STATIC 1

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

	// Clear framebuffer and zbuffer
	glClearColor(0.0, 0.0, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Create quad that covers screen to render to
	float quadVertices[] = {-1.0, -1.0,
							1.0, -1.0,
							-1.0, 1.0,
							1.0, 1.0 };
	// Use shader
	glUseProgram(shaderProgram);
	glUniform1fv(glGetUniformLocation(shaderProgram, "quadVertices"), 8, quadVertices);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
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
bool initGL(int *argc, char **argv)
{
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK){
		throw std::exception("Failed to initialise GLEW\n");
		//printf("FAILED TO initialize GLEW");
	}
	else {
		//printf("SUCCESSEDED IN Initialising GLEW\n");
	}

	if (!glewIsSupported("GL_ARB_vertex_buffer_object"))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		return false;
	}

	glViewport(0.0f, 0.0f, WINDOW_WIDTH, WINDOW_HEIGHT);
	
	// Initialize projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0,1.0,-1.0);

	// Initialize model view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Initialize clear color
	glClearColor(0.1,0.1,0.2,0);

	// Enable texturing
	glEnable(GL_TEXTURE_2D);

	// Set bledning
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Check for error
	GLenum error = glGetError();
	if (error != GL_NO_ERROR)
	{
		printf("Error \n");
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
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

	glutCreateWindow("Compute Stable Fluids");
	glutDisplayFunc(display);

	if (!initGL(&argc, argv)){
		printf("Unable to initialize GLEW\n");
	}

	void *fluidVelocityData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidVelocityType));
	void *fluidPressureData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidPressureType));
	initCuda(fluidVelocityData, fluidPressureData, volumeSize);
	
	//initGL(&argc, argv);
	
	simulateFluid();

	while (true)
	{
		display();
	}
	
	return 0;
}
