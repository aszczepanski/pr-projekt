#include "multiplicator1.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include "cuda_exceptions.h"
#include "array_manager.h"

__global__ void MatrixMulKernel_1(const float* Ad, const float* Bd, float* Cd, const int WIDTH) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float C_local;
	
	for (int i=0; i<WIDTH/blockDim.y; i++) {
		for (int j=0; j<WIDTH/blockDim.x; j++) {
			C_local = 0.0f;
			for (int k = 0; k < WIDTH; ++k) {
				float A_d_element = Ad[i*WIDTH*blockDim.y + ty*WIDTH + k];
				float B_d_element = Bd[j*blockDim.y + k*WIDTH + tx];
				C_local += A_d_element * B_d_element;
			}

			Cd[i*WIDTH*blockDim.y + j*blockDim.y + ty*WIDTH + tx] = C_local;
		}
	}
}

void Multiplicator1::launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager) {
	clearErrorFlag();

	// Launch a kernel on the GPU with one thread for each element.
	dim3 gridDim;
	dim3 blockDim(blockSize, blockSize);
	MatrixMulKernel_1<<<gridDim, blockDim>>>(
		arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arrayManager->pointerToDev_C(), arraySize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatrixMulKernel_1 launch failed\n");
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}