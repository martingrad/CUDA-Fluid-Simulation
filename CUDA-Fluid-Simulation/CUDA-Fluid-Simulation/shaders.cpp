/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
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

	void main()
	{
		out_Color = texture(velocityTex, vec3(1));
		//out_Color = vec4(0.5,0.5,0.7,1.0);
	}
);
