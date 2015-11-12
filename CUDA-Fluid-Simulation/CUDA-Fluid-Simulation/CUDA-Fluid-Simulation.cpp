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

cudaExtent volumeSize = make_cudaExtent(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Y);
const int NUMBER_OF_GRID_CELLS = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z;

StopWatchInterface *timer = NULL;
static int fpsCount = 0;
int fpsLimit = 1;

GLuint shaderProgram = NULL;

// OpenGL simulation data textures
GLuint glTex_velocity;
GLuint glTex_velocityTest;

cudaGraphicsResource *cuda_image_resource;
cudaArray            *cuda_image_array;

extern "C" void advectVelocity();
extern "C" void initCuda(void *fluidData_velocity, void* fluidData_pressure, cudaExtent volumeSize);
extern "C" void launch_kernel(struct cudaArray *cuda_image_array, dim3 texture_dim, float testFloatX, float testFloatY, float testFloatZ);

float testFloatX = 0.0f;
float testFloatY = 0.0f;
float testFloatZ = 0.0f;

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
	glUniform1i(glGetUniformLocation(shaderProgram, "velocityTex"), 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, glTex_velocityTest);
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

	// Trigger next frame
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

// Check Texture
void checkTex()
{
	int numElements = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z * 4;
	float *data = new float[numElements];

	glBindTexture(GL_TEXTURE_3D, glTex_velocityTest);
	{
		glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, data);
	}
	glBindTexture(GL_TEXTURE_3D, 0);

	bool fail = false;
	for (int i = 0; i < numElements && !fail; i++)
	{
		if (data[i] != 1.0f)
		{
			std::cerr << "Not 1.0f, failed writing to texture. Value is: " << data[i] << std::endl;
			fail = true;
		}
	}
	if (!fail)
	{
		std::cerr << "All Elements == 1.0f, texture write successful" << std::endl;
	}

	delete[] data;
}

void initCuda()
{
	float4* texels;
	texels = new float4[VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z];
	//texels = (float4*)malloc(VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z * sizeof(float4));
	for (int i = 0; i < VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z; ++i)
	{
		texels[i] = make_float4(1.0, 0.0, 0.0, 1.0);
	}

	glGenTextures(1, &glTex_velocityTest);
	glBindTexture(GL_TEXTURE_3D, glTex_velocityTest);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Z, 0, GL_RGBA, GL_FLOAT, texels);

	// Unbind texture
	glBindTexture(GL_TEXTURE_3D, 0);

	//CUT_CHECK_ERROR_GL();

	// register Image (texture) to CUDA Resource
	cudaGraphicsGLRegisterImage(&cuda_image_resource, glTex_velocityTest, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	// map CUDA resource
	cudaGraphicsMapResources(1, &cuda_image_resource, 0);

	//Get mapped array
	cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, cuda_image_resource, 0, 0);
	dim3 textureDim = dim3(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Z);
	launch_kernel(cuda_image_array, textureDim, testFloatX, testFloatY, testFloatZ);

	cudaGraphicsUnmapResources(1, &cuda_image_resource, 0);

	//checkTex();

	cudaGraphicsUnregisterResource(cuda_image_resource);
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

	//fluidVelocityType* fluidVelocityData = (fluidVelocityType*)malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidVelocityType));
	fluidVelocityType* fluidVelocityData = new fluidVelocityType[NUMBER_OF_GRID_CELLS];
	void* fluidPressureData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidPressureType));

	// Velocity data
	for (int i = 0; i < volumeSize.width * volumeSize.width * volumeSize.depth; ++i)
	{
		fluidVelocityData[i] = (fluidVelocityType)make_float4(1.0, 1.0, 1.0, 1.0);
	}

	//initCuda(fluidVelocityData, fluidPressureData, volumeSize);

	initCuda();

	//glDeleteTextures(1, &glTex_velocityTest);

	//cutilDeviceReset();

	simulateFluid();

	glutMainLoop();
	
	return 0;
}
