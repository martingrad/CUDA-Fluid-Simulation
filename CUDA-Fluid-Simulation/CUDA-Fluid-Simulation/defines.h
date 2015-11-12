/*
 *
 *
 */
 
#ifndef DEFINES_H
#define DEFINES_H

#define VOLUME_SIZE_X 256
#define VOLUME_SIZE_Y 256
#define VOLUME_SIZE_Z 256

#define THREADS_PER_BLOCK 512		// 8 * 8 * 8
#define NUMBER_OF_BLOCKS 512		// 8 * 8 * 8

#define WINDOW_HEIGHT	512
#define WINDOW_WIDTH	512

typedef unsigned char fluidPressureType;	// 0-255
typedef float4 fluidVelocityType;

#endif
