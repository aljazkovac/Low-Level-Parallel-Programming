#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_heatmap.h"
#include <stdio.h>

__global__ void creationKernel(int *desiredX, int *desiredY, int **heatmap)
{
  int i = threadIdx.x;

  int x = desiredX[i];
  int y = desiredY[i];

  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {}
  else {
    atomicAdd(&heatmap[y][x], 40);
  }
}

__global__ void scalingKernel(int **heatmap, int **scaled_heatmap)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
  int value = heatmap[y][x];
  for (int cellY = 0; cellY < CELLSIZE; cellY++)
    {
      for (int cellX = 0; cellX < CELLSIZE; cellX++)
	{
	  scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
	}
    }
}

/* __global__ void blurKernel() */
/* { */
  
/* } */

// Calculates and updates x/y positions, checks if agent has reached destination -> destReached
cudaError_t updateHeatmapCuda(int *desiredX, int *desiredY, int **heatmap, int **scaled_heatmap)
{
  int agents_size = agents.size();
  cudaError_t cudaStatus;
  /* int *dev_desiredX; */
  /* int *dev_desiredY; */
  /* int **dev_heatmap; */
  /* int **dev_scaled_heatmap; */
  int NUM_BLOCKS = 2048;
  int THREADS_PER_BLOCK = 256;
  int size = NUM_BLOCKS * THREADS_PER_BLOCK;
  
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers
  cudaStatus = cudaMemcpy(dev_desiredX, desiredX, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_desiredY, desiredY, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_heatmap, heatmap, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_scaled_heatmap, scaled_heatmap, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}

  // Launch Kernel on the GPU with one thread for each element
  creationKernel <<<1, size>>> (dev_desiredX, dev_desiredX, dev_heatmap);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "tickKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  // Synchronize
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error;
  }
  
  dim3 threadsPerBlock(16,16); // 256 threads per block
  dim3 numBlocks(SIZE/threadsPerBlock.x, SIZE/threadsPerBlock.y);
  checkAndScaleKernel <<<numBlocks, threadsPerBlock>>> (dev_heatmap, dev_scaled_heatmap);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "tickKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  // Synchronize
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus); goto Error;
  }

  // Copy data from device to host
  cudaStatus = cudaMemcpy(desiredX, dev_desiredX, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(desiredY, dev_desiredY, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(heatmap, dev_heatmap, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
  cudaStatus = cudaMemcpy(heatmap, dev_scaled_heatmap, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}

 Error:
  /* cudaFree(dev_desiredX); */
  /* cudaFree(dev_desiredY); */
  /* cudaFree(dev_heatmap); */
  /* cudaFree(dev_scaled_heatmap); */
  /* cudaFree(dev_destReached); */
  if (cudaStatus != 0){
    fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
  }
  /* else{ */
  /* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
  /* } */

  return cudaStatus;
}

cudaError_t allocCuda(int size);
{
  // Allocate GPU buffers for agents
  cudaStatus = cudaMalloc((void **)&dev_desiredX, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_desiredY, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_heatmap, SIZE * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_scaled_heatmap, SIZE * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  
  Error:
  /* cudaFree(dev_desiredX); */
  /* cudaFree(dev_desiredY); */
  /* cudaFree(dev_heatmap); */
  /* cudaFree(dev_scaled_heatmap); */
  /* cudaFree(dev_destReached); */
  if (cudaStatus != 0){
    fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
  }
  /* else{ */
  /* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
  /* } */

  return cudaStatus;
}
