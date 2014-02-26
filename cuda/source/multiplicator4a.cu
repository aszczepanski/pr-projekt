#include "multiplicator4a.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_exceptions.h"
#include "array_manager.h"

template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_4a(float *C, const float *A, const float *B, const int arraySize) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = arraySize * BLOCK_SIZE * by;
    int aEnd   = aBegin + arraySize - 1;
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * arraySize;

    float Csub = 0.0f;
	
	float fetchA, fetchB;
	fetchA = A[aBegin + arraySize * ty + tx];
    fetchB = B[bBegin + arraySize * ty + tx];

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[ty][tx] = fetchA;
		Bs[ty][tx] = fetchB;

        __syncthreads();

		fetchA = A[a + aStep + arraySize * ty + tx];
		fetchB = B[b + bStep + arraySize * ty + tx];

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = arraySize * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + arraySize * ty + tx] = Csub;
}

void Multiplicator4a::launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager) {
	clearErrorFlag();
	
	// Setup execution parameters
    dim3 threads(blockSize, blockSize);
    dim3 grid(ceil((float)arraySize/(float)blockSize), ceil((float)arraySize/(float)blockSize));

	// Execute the kernel
	if (blockSize == 8) {
        MatrixMulKernel_4a<8><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
	} else if (blockSize == 16) {
        MatrixMulKernel_4a<16><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
    } else if (blockSize == 22) {
        MatrixMulKernel_4a<22><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
    }
	
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMulCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}