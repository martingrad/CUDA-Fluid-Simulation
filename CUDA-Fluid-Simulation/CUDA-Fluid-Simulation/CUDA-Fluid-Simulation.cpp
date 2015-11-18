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
#include "glm\glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

cudaExtent volumeSize = make_cudaExtent(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Y);
const int NUMBER_OF_GRID_CELLS = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z;

StopWatchInterface *timer = NULL;
static int fpsCount = 0;
int fpsLimit = 1;

GLuint shaderProgram = NULL;

// Simulation data textures
GLuint glTex_velocity;

// GL transformation matrices
glm::mat4 ProjectionMatrix;
glm::mat4 ModelviewMatrix;
glm::mat4 ViewMatrix;
glm::mat4 ModelviewProjection;

// Rendering data
glm::vec3 EyePosition;
static float FieldOfView = 0.7f;

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
	// Call update/simulate function...?

	// [TEMP] update eye position
	/*EyePosition += glm::vec3(1.0, 1.0, 1.0);
	ViewMatrix = glm::lookAt(EyePosition, glm::vec3(0.0), glm::vec3(0.0, 1.0, 0.0));*/
	
	sdkStartTimer(&timer);

	// Clear framebuffer and zbuffer
	glClearColor(0.5, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	// Create quad that covers screen to render to
	float quadVertices[] = {-1.0, -1.0,
							1.0, -1.0,
							-1.0, 1.0,
							1.0, 1.0 };

	// Use shader
	glUseProgram(shaderProgram);

	// Bind uniforms
	
	//  Quad data
	glUniform1fv(glGetUniformLocation(shaderProgram, "quadVertices"), 8, quadVertices);
	//  3D simulation data textures
	glUniform1i(glGetUniformLocation(shaderProgram, "Density"), 0);

	//  Transformation matrices
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "ModelviewProjection"), 1, 0, (GLfloat*)&ModelviewProjection);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "Modelview"), 1, 0, (GLfloat*)&ModelviewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "ViewMatrix"), 1, 0, (GLfloat*)&ViewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "ProjectionMatrix"), 1, 0, (GLfloat*)&ProjectionMatrix);
	//  Rendering data
	glUniform3f(glGetUniformLocation(shaderProgram, "EyePosition"), EyePosition.x, EyePosition.y, EyePosition.z);
	glUniform2f(glGetUniformLocation(shaderProgram, "WindowSize"), WINDOW_WIDTH, WINDOW_HEIGHT);
	float focalLength = 1.0f / std::tan(FieldOfView / 2);
	glUniform1f(glGetUniformLocation(shaderProgram, "FocalLength"), focalLength);
	glm::vec4 rayOrigin(transpose(ModelviewMatrix) * glm::vec4(EyePosition, 0.0));
	glUniform3f(glGetUniformLocation(shaderProgram, "RayOrigin"), rayOrigin.x, rayOrigin.x, rayOrigin.x);
	
	// Set active texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, glTex_velocity);
	
	// Simple draw call
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glUseProgram(0);

	// Finish timing before swap buffers to avoid refresh sync
	sdkStopTimer(&timer);
	glutSwapBuffers();

	/*fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Cuda/GL Stable Fluids (%d x %d): %3.1f fps", WINDOW_HEIGHT, WINDOW_HEIGHT, ifps);
		glutSetWindowTitle(fps);
		fpsCount = 0;
		fpsLimit = (int)MAX(ifps, 1.f);
		sdkResetTimer(&timer);
	}*/

	// Trigger next frame
	glutPostRedisplay();
}

/*
* InitGL
*/
bool initGL(int *argc, char **argv)
{

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Init transfomation matrices
	glm::vec3 up(0, 1, 0);
	glm::vec3 target(0.0, 0.0, 0.0);
	EyePosition = glm::vec3(9.0, 9.0, 9.0);
	
	ViewMatrix = glm::lookAt(EyePosition, target, up);
	ProjectionMatrix = glm::perspective(60.0f, (float)WINDOW_WIDTH / WINDOW_HEIGHT, 1.0f, 100.0f);
	ModelviewProjection = ProjectionMatrix * ViewMatrix; // TODO: model part?

	glm::mat4 modelMatrix = glm::mat4(); // Temporary modelMatrix. TODO: model part?
	ModelviewMatrix = ViewMatrix * modelMatrix;
	
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK){
		throw std::exception("Failed to initialise GLEW\n");
		//printf("FAILED TO initialize GLEW");
	}
	else {
		//printf("SUCCEEDED IN initialising GLEW\n");
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
	GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);

	glShaderSource(vertexShader, 1, &vertexShaderContents, 0);
	glShaderSource(fragmentShader, 1, &fragmentShaderContents, 0);
	glShaderSource(geometryShader, 1, &geometryShaderContents, 0);

	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);
	glCompileShader(geometryShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	//glAttachShader(program, geometryShader);

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

	gluPerspective(75, 1, 0, 100);

	return true;
}

// Check Texture
void checkTex()
{
	int numElements = VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z * 4;
	float *data = new float[numElements];

	glBindTexture(GL_TEXTURE_3D, glTex_velocity);
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
	
	// Velocity texture initial data
	for (int i = 0; i < VOLUME_SIZE_X * VOLUME_SIZE_Y * VOLUME_SIZE_Z; ++i)
	{
		texels[i] = make_float4(0.01, 0.0, 0.0, 1.0);
	}

	glGenTextures(1, &glTex_velocity);
	glBindTexture(GL_TEXTURE_3D, glTex_velocity);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Z, 0, GL_RGBA, GL_FLOAT, texels);

	// Unbind texture
	glBindTexture(GL_TEXTURE_3D, 0);

	//CUT_CHECK_ERROR_GL();

	// Register Image (texture) to CUDA Resource
	cudaGraphicsGLRegisterImage(&cuda_image_resource, glTex_velocity, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	// Map CUDA resource
	cudaGraphicsMapResources(1, &cuda_image_resource, 0);

	// Get mapped array
	cudaGraphicsSubResourceGetMappedArray(&cuda_image_array, cuda_image_resource, 0, 0);
	dim3 textureDim = dim3(VOLUME_SIZE_X, VOLUME_SIZE_Y, VOLUME_SIZE_Z);
	launch_kernel(cuda_image_array, textureDim, testFloatX, testFloatY, testFloatZ);

	cudaGraphicsUnmapResources(1, &cuda_image_resource, 0);

	//checkTex();

	cudaGraphicsUnregisterResource(cuda_image_resource);

	delete texels;
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

	fluidVelocityType* fluidVelocityData = new fluidVelocityType[NUMBER_OF_GRID_CELLS];
	void* fluidPressureData = malloc(NUMBER_OF_GRID_CELLS * sizeof(fluidPressureType));

	//// Create velocity data
	//for (int i = 0; i < volumeSize.width * volumeSize.width * volumeSize.depth; ++i)
	//{
	//	fluidVelocityData[i] = (fluidVelocityType)make_float4(0.01, 1.0, 1.0, 1.0);
	//}

	initCuda();

	//glDeleteTextures(1, &glTex_velocity);
	//cutilDeviceReset();

	simulateFluid();

	// Delete volume data from (regular) RAM
	delete fluidVelocityData;
	delete fluidPressureData;

	glutMainLoop();
	
	return 0;
}
