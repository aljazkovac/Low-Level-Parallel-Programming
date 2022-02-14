#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t tickCuda(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached, int NUM_BLOCKS, int THREADS_PER_BLOCK);
__global__ void tickKernel(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached);
