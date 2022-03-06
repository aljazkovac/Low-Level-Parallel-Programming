#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include "ped_model.h"

cudaError_t allocCuda(int size);
cudaError_t updateHeatmapCuda(int *desiredX, int *desiredY, int *hm, int *shm, int *bhm, int agent_size);
__global__ void creationKernel(int *desiredX, int *desiredY, int *hm);
__global__ void scalingKernel(int *hm, int *shm);
