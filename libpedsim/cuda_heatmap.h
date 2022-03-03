#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include "ped_model.h"


int *dev_desiredX;
int *dev_desiredY;
int **dev_heatmap;
int **dev_scaled_heatmap;

cudaError_t allocCuda();
cudaError_t updateHeatmapCuda(int *desiredX, int *desiredY, int **heatmap);
__global__ void creationKernel(int *desiredX, int *desiredY, int **heatmap);
