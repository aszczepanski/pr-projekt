#include "multiplicator2.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_exceptions.h"
#include "array_manager.h"

__global__ void MatrixMulKernel_2(const float* Ad, const float* Bd, float* Cd, int WIDTH) {
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	float C_local = 0.0f;

	for (int k = 0; k < WIDTH; ++k)
		C_local += Ad[Row*WIDTH + k] * Bd[k*WIDTH + Col];
	
	Cd[Row*WIDTH + Col] = C_local;
}  

void Multiplicator2::launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager) {
	clearErrorFlag();
	
	dim3 gridDim(ceil((float)arraySize/(float)blockSize), ceil((float)arraySize/(float)blockSize));
	dim3 blockDim(blockSize, blockSize);

	MatrixMulKernel_2<<<gridDim, blockDim>>>(
		arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arrayManager->pointerToDev_C(), arraySize);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatrixMulKernel_2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}