#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_heatmap.h"
#include <stdio.h>

__global__ void creationKernel(int *desiredX, int *desiredY, int *hm)
{
  //int i = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int x = desiredX[i];
  int y = desiredY[i];

  if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {}
  else {
    atomicAdd(&hm[y*SIZE + x], 40);
  }
}

__global__ void scalingKernel(int *hm, int *shm)
{
  __shared__ int shm_temp[16 * 16 * CELLSIZE * CELLSIZE];
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int x_temp = threadIdx.x;
  int y_temp = threadIdx.y;

  int SMALL_SCALED_SIZE = 16 * CELLSIZE;

  hm[y*SIZE + x] = (int)round(hm[y*SIZE + x] * 0.80);
  hm[y*SIZE + x] = hm[y*SIZE + x] < 255 ? hm[y*SIZE + x] : 255;
  int value = hm[y*SIZE + x];
  for (int cellY = 0; cellY < CELLSIZE; cellY++)
    {
      for (int cellX = 0; cellX < CELLSIZE; cellX++)
	{
	  shm_temp[(y_temp * CELLSIZE + cellY)*SMALL_SCALED_SIZE + (x_temp * CELLSIZE + cellX)] = value;
	}
    }
  __syncthreads();
  for (int cellY = 0; cellY < CELLSIZE; cellY++)
    {
      for (int cellX = 0; cellX < CELLSIZE; cellX++)
	{
	  shm[(y * CELLSIZE + cellY)*SCALED_SIZE + (x * CELLSIZE + cellX)] = shm_temp[(y_temp * CELLSIZE + cellY)*SMALL_SCALED_SIZE + (x_temp * CELLSIZE + cellX)];
	}
    }
}

__global__ void blurKernel(int *shm, int *bhm)
{
  const int w[5][5] = {
    { 1, 4, 7, 4, 1 },
    { 4, 16, 26, 16, 4 },
    { 7, 26, 41, 26, 7 },
    { 4, 16, 26, 16, 4 },
    { 1, 4, 7, 4, 1 }
  };

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

#define WEIGHTSUM 273
  // Apply gaussian blurfilter

  if (x < 2 || x > SCALED_SIZE - 2) {
    return;
  }
  if (y < 2 || y > SCALED_SIZE - 2) {
    return;
  }
  
  int sum = 0;
  for (int k = -2; k < 3; k++)
    {
      for (int l = -2; l < 3; l++)
	{
	  // sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
	  sum += w[2 + k][2 + l] * shm[(y + k)*SCALED_SIZE + (x + l)];
	}
    }
  int value = sum / WEIGHTSUM;
  bhm[y*SCALED_SIZE + x] = 0x00FF0000 | value << 24;
}

// Calculates and updates x/y positions, checks if agent has reached destination -> destReached
cudaError_t updateHeatmapCuda(int *desiredX, int *desiredY, int *hm, int *shm, int *bhm, int agents_size)
{
  cudaError_t cudaStatus;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;


  // int NUM_BLOCKS = 2048;
  // int THREADS_PER_BLOCK = 512;
  //  int size = NUM_BLOCKS * THREADS_PER_BLOCK;

  int *dev_desiredX;
  int *dev_desiredY;
  int *dev_hm;
  int *dev_shm;
  int *dev_bhm;

  dim3 threadsPerBlock(16,16); // 256 threads per block
  dim3 numBlocks(SIZE/threadsPerBlock.x, SIZE/threadsPerBlock.y);
  
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
    goto Error;
  }

  // Allocate GPU buffers for agents
  cudaStatus = cudaMalloc((void **)&dev_desiredX, agents_size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "dx cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_desiredY, agents_size * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_hm, SIZE * SIZE * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  cudaStatus = cudaMalloc((void **)&dev_bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}

  // Copy input vectors from host memory to GPU buffers
  cudaStatus = cudaMemcpy(dev_desiredX, desiredX, agents_size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy dx to device failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_desiredY, desiredY, agents_size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy dy to failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_hm, hm, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy hm to device failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_shm, shm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy shm to device failed!"); goto Error;}
  cudaStatus = cudaMemcpy(dev_bhm, bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy shm to device failed!"); goto Error;}

  

  // Launch Kernel on the GPU with one thread for each element
  // Set 10 blocks for hugeScenario.xml
  cudaEventRecord(start);
  creationKernel <<<10, agents_size/10>>> (dev_desiredX, dev_desiredY, dev_hm);
  cudaEventRecord(stop);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "creationKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("creationKernel ms: %f\n", milliseconds);

  // Synchronize
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching creationKernel!\n", cudaStatus); goto Error;
  }

  cudaEventRecord(start);
  scalingKernel <<<numBlocks, threadsPerBlock>>> (dev_hm, dev_shm);
  cudaEventRecord(stop);
  
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "scalingKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("scaleKernel ms: %f\n", milliseconds);

  // Synchronize
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching scalingKernel!\n", cudaStatus); goto Error;
  }

  // Blur filter
  cudaEventRecord(start);
  blurKernel <<<numBlocks, threadsPerBlock>>> (dev_shm, dev_bhm);
  cudaEventRecord(stop);
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "blurKernel launch failed: %s\n", cudaGetErrorString(cudaStatus)); goto Error;}

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("blurKernel ms: %f\n", milliseconds);

  // Copy data from device to host
  cudaStatus = cudaMemcpy(hm, dev_hm, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy hm to host failed!"); goto Error;}
  cudaStatus = cudaMemcpy(shm, dev_shm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy shm to host failed!"); goto Error;}
  cudaStatus = cudaMemcpy(bhm, dev_bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy shm to host failed!"); goto Error;}

 Error:
  cudaFree(dev_desiredX);
  cudaFree(dev_desiredY);
  cudaFree(dev_hm);
  cudaFree(dev_shm);
  cudaFree(dev_bhm);
  if (cudaStatus != 0){
    fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
  }
  /* else{ */
  /* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
  /* } */

  return cudaStatus;
}

// cudaError_t allocCuda(int size)
// {
//   cudaError_t cudaStatus;

//   // Allocate GPU buffers for agents
//   cudaStatus = cudaMalloc((void **)&dev_desiredX, size * sizeof(int));
//   if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
//   cudaStatus = cudaMalloc((void **)&dev_desiredY, size * sizeof(int));
//   if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  // cudaStatus = cudaMalloc((void **)&dev_hm, SIZE * sizeof(int));
  // if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
//   cudaStatus = cudaMalloc((void **)&dev_shm, SIZE * sizeof(int));
//   if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMalloc failed!"); goto Error;}
  
//   Error:
//   /* cudaFree(dev_desiredX); */
//   /* cudaFree(dev_desiredY); */
//   /* cudaFree(dev_hm); */
//   /* cudaFree(dev_shm); */
//   /* cudaFree(dev_destReached); */
//   if (cudaStatus != 0){
//     fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
//   }
//   /* else{ */
//   /* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
//   /* } */

//   return cudaStatus;
// }
