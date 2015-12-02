/*
*
*/

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShaderContents = STRINGIFY(
#version 330 core
	\n
	uniform float quadVertices[8];
uniform sampler3D velocityTex;
void main()
{
	// Draw quad from supplied vertex array
	gl_Position = vec4(quadVertices[gl_VertexID * 2] - 1, quadVertices[gl_VertexID * 2 + 1] - 1, 0.0, 1.0);
}
);

// fragment shader
const char *fragmentShaderContents = STRINGIFY(
	
	#version 330 core
		\n
		uniform sampler3D velocityTex; //texture unit 0
	out vec4 out_Color;

	struct Ray
	{
		vec3 startingPosition;
		vec3 direction;
		float opacity;
		float stepSize;
	} ray;

	void main()
	{
		ray.startingPosition = vec3(gl_FragCoord);

		out_Color = texture(velocityTex, vec3(0.0));
	}

);

