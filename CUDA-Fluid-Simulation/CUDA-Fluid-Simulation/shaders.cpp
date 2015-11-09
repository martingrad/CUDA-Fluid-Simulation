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
	#version 330 core \n
	uniform float quadVertices[8];
	void main()
	{
		float a[5] = float[](3.4, 4.2, 5.0, 5.2, 1.1);
		gl_Position = vec4(quadVertices[gl_VertexID * 2], quadVertices[gl_VertexID * 2 + 1], 0.0, 1.0);
	}
);

// fragment shader
const char *fragmentShaderContents = STRINGIFY(
	void main()
	{
		gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
);
