/*
 *
 *
 */
 
#ifndef DEFINES_H
#define DEFINES_H

#define VOLUME_SIZE_X 64
#define VOLUME_SIZE_Y 64
#define VOLUME_SIZE_Z 64

#define THREADS_PER_BLOCK 512		// 8 * 8 * 8
#define NUMBER_OF_BLOCKS 512		// 8 * 8 * 8

typedef unsigned char fluidPressureType;	// 0-255
typedef float4 fluidVelocityType;

#endif
