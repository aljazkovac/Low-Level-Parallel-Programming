#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_tick.h"
#include <stdio.h>

__global__ void tickKernel(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float diffX = destXarray[i] - xArray[i];
  float diffY = destYarray[i] - yArray[i];

  float length = sqrt(diffX*diffX + diffY*diffY);
  destReached[i] = length < destRarray[i];

  xArray[i] = (int) round(xArray[i] + diffX/length);
  yArray[i] = (int) round(yArray[i] + diffY/length);
}

// Calculates and updates x/y positions, checks if agent has reached destination -> destReached
cudaError_t tickCuda(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached, int NUM_BLOCKS, int THREADS_PER_BLOCK)
{
  cudaError_t cudaStatus;
  int *dev_xArray;
  int *dev_yArray;
  float *dev_destXarray;
  float *dev_destYarray;
  float *dev_destRarray;
  int *dev_destReached;
  int size = NUM_BLOCKS * THREADS_PER_BLOCK;
  
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
    goto Error;
  }

  // Allocate GPU buffers for vectors
  cudaStatus = cudaMalloc((void**)&dev_xArray, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void**)&dev_yArray, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void**)&dev_destXarray, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void**)&dev_destYarray, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void**)&dev_destRarray, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void**)&dev_destReached, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}

  // Copy input vectors from host memory to GPU buffers
  cudaStatus = cudaMemcpy(dev_xArray, xArray, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_yArray, yArray, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_destXarray, destXarray, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_destYarray, destYarray, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_destRarray, destRarray, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_destReached, destReached, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}

  // Launch Kernel on the GPU with one thread for each element
  tickKernel <<<NUM_BLOCKS, THREADS_PER_BLOCK>>> (dev_xArray, dev_yArray, dev_destXarray, dev_destYarray, dev_destRarray, dev_destReached);
  
  // Check if kernel succeded
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "tickKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error;
  }

  // Copy data from device to host
  cudaStatus = cudaMemcpy(xArray, dev_xArray, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(yArray, dev_yArray, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(destReached, dev_destReached, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
Error:
	/* cudaFree(dev_xArray); */
	/* cudaFree(dev_yArray); */
	/* cudaFree(dev_destXarray); */
	/* cudaFree(dev_destYarray); */
	/* cudaFree(dev_destRarray); */
	/* cudaFree(dev_destReached); */
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	/* else{ */
	/* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
	/* } */

	return cudaStatus;
}
