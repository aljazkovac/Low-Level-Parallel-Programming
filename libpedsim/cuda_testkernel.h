#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int cuda_test();
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void  addKernel(int *c, const int *a, const int *b);
cudaError_t tickCuda(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached, int NUM_BLOCKS, int THREADS_PER_BLOCK);
__global__ void tickKernel(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached);
__global__ void tickKernel2(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached);
