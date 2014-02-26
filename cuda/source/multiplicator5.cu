#include "multiplicator5.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_exceptions.h"
#include "array_manager.h"

template <int BLOCK_SIZE> __global__ void
MatrixMulKernel_5(float *C, const float *A, const float *B, const int arraySize) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = 2 * arraySize * BLOCK_SIZE * by;
    int aEnd   = aBegin + arraySize - 1;
    int aStep  = BLOCK_SIZE;

    int bBegin = 2 * BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * arraySize;

	float Csub00=0.0f, Csub01=0.0f, Csub10=0.0f, Csub11=0.0f;
	float fetchA0, fetchA1;
	float fetchB0, fetchB1;
	fetchA0 = A[aBegin + arraySize * 2*ty + tx];
	fetchA1 = A[aBegin + arraySize * (2*ty+1) + tx];
	fetchB0 = B[bBegin + arraySize * ty + 2*tx];
	fetchB1 = B[bBegin + arraySize * ty + 2*tx+1];

    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {
        __shared__ float As[2 * BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][2 * BLOCK_SIZE];
		
		As[2*ty+0][tx] = fetchA0;
		As[2*ty+1][tx] = fetchA1;
		Bs[ty][2*tx+0] = fetchB0;
		Bs[ty][2*tx+1] = fetchB1;

		__syncthreads();
		
		fetchA0 = A[a + aStep + arraySize * 2*ty + tx];
		fetchA1 = A[a + aStep + arraySize * (2*ty+1) + tx];
		fetchB0 = B[b + bStep + arraySize * ty + 2*tx];
		fetchB1 = B[b + bStep + arraySize * ty + 2*tx+1];
		
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub00 += As[2*ty][k] * Bs[k][2*tx];
		}
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub01 += As[2*ty][k] * Bs[k][2*tx+1];
		}
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub10 += As[2*ty+1][k] * Bs[k][2*tx];
		}
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub11 += As[2*ty+1][k] * Bs[k][2*tx+1];
		}
		
        __syncthreads();
    }

	int c = 2 * arraySize * BLOCK_SIZE * by + 2 * BLOCK_SIZE * bx;
	C[c + arraySize * (2*ty) + 2*tx] = Csub00;
	C[c + arraySize * (2*ty) + 2*tx+1] = Csub01;
	C[c + arraySize * (2*ty+1) + 2*tx] = Csub10;
	C[c + arraySize * (2*ty+1) + 2*tx+1] = Csub11;
	
}

void Multiplicator5::launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager) {
	clearErrorFlag();

	// Setup execution parameters
    dim3 threads(blockSize, blockSize);
    dim3 grid(ceil((float)arraySize/(2.0f*(float)blockSize)), ceil((float)arraySize/(2.0f*(float)blockSize)));

	// Execute the kernel
	if (blockSize == 8) {
        MatrixMulKernel_5<8><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
	} else if (blockSize == 16) {
        MatrixMulKernel_5<16><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
    } else if (blockSize == 22) {
        MatrixMulKernel_5<22><<< grid, threads >>>(
			arrayManager->pointerToDev_C(), arrayManager->pointerToDev_A(), arrayManager->pointerToDev_B(), arraySize);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMulCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}