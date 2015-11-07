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

	void main()
	{

		gl_Position = vec4(gl_Vertex.xyz, 1.0);
	}
);

// fragment shader
const char *fragmentShaderContents = STRINGIFY(

	void main()
	{
		gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
	}
);
