#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

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

__global__ void tickKernel2(int *xArray, int *yArray, float *destXarray, float *destYarray, float *destRarray, int *destReached)
{
  int i = threadIdx.x;
  
  float diffX = destXarray[i] - xArray[i];
  float diffY = destYarray[i] - yArray[i];
  
  float length = sqrt(diffX*diffX + diffY*diffY);
  destReached[i] = length < destRarray[i];

  xArray[i] = (int) round(xArray[i] + diffX/length);
  yArray[i] = (int) round(yArray[i] + diffY/length);
}

int cuda_test()
{
    static int tested = 0;

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

    if (tested == 1)
        return 0;
    tested = 1;

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
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
  //  tickKernel2 <<<1, NUM_BLOCKS * THREADS_PER_BLOCK>>> (dev_xArray, dev_yArray, dev_destXarray, dev_destYarray, dev_destRarray, dev_destReached);
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
  /* cudaStatus = cudaMemcpy(destXarray, dev_destXarray, size * sizeof(int), cudaMemcpyDeviceToHost); */
  /* if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;} */
  /* cudaStatus = cudaMemcpy(destYarray, dev_destYarray, size * sizeof(int), cudaMemcpyDeviceToHost); */
  /* if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;} */
  /* cudaStatus = cudaMemcpy(destRarray, dev_destRarray, size * sizeof(int), cudaMemcpyDeviceToHost); */
  /* if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;} */
  cudaStatus = cudaMemcpy(destReached, dev_destReached, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {fprintf(stderr, "cudaMemcpy failed!"); goto Error;}
Error:
	cudaFree(dev_xArray);
	cudaFree(dev_yArray);
	cudaFree(dev_destXarray);
	cudaFree(dev_destYarray);
	cudaFree(dev_destRarray);
	cudaFree(dev_destReached);
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	/* else{ */
	/* 	fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing */
	/* } */

	return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		fprintf(stderr, "%s.\n", cudaGetErrorString(cudaGetLastError()));
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	else
	{
		//fprintf(stderr, "Cuda launch succeeded! \n");
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	if (cudaStatus != 0){
		fprintf(stderr, "Cuda does not seem to be working properly.\n"); // This is not a good thing
	}
	else{
		fprintf(stderr, "Cuda functionality test succeeded.\n"); // This is a good thing
	}

	return cudaStatus;
}
