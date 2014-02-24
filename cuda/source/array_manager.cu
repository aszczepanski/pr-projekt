#include "array_manager.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda_exceptions.h"

void ArrayManager::allocateAndInitializeArrays(const size_t arraySize) {
    memorySizeForArray = sizeof(float) * arraySize * arraySize;

    allocateHostArrays();
	
	initializeArrayWithConstValue(host_C, arraySize, 0.0f);
	initializeArrayWithOnesOnDiagonal(host_A, arraySize);
	initializeArrayWithSumIJ(host_B, arraySize);

	allocateDeviceArrays();
}

void ArrayManager::initializeArrayWithOnesOnDiagonal(float* array, size_t arraySize) {
	for (size_t i=0; i<arraySize; i++) {
		for (size_t j=0; j<arraySize; j++) {			
			array[i*arraySize + j] = ((i==j) ? 1.0f : 0.0f);
		}
	}
}

void ArrayManager::initializeArrayWithSumIJ(float* array, size_t arraySize) {
	for (size_t i=0; i<arraySize; i++) {
		for (size_t j=0; j<arraySize; j++) {
			array[i*arraySize + j] = (float)(i+j);
		}
	}
}

void ArrayManager::initializeArrayWithConstValue(float* array, size_t size, const float value) {
    for (size_t i = 0; i < size; ++i) {
        array[i] = value;
    }
}

void ArrayManager::allocateHostArrays() {
	host_A = (float*)malloc(memorySizeForArray);
    host_B = (float*)malloc(memorySizeForArray);
    host_C = (float*)malloc(memorySizeForArray);
}

void ArrayManager::allocateDeviceArrays() {
	cudaStatus = cudaMalloc((void**) &dev_A, memorySizeForArray);
	if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }

    cudaStatus = cudaMalloc((void**) &dev_B, memorySizeForArray);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }

    cudaStatus = cudaMalloc((void**) &dev_C, memorySizeForArray);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}

void ArrayManager::sendDataToDevice() {
    cudaStatus = cudaMemcpy(dev_A, host_A, memorySizeForArray, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy (d_A,h_A) returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }

    cudaStatus = cudaMemcpy(dev_B, host_B, memorySizeForArray, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy (d_B,h_B) returned error code %d, line(%d)\n", cudaStatus, __LINE__);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}

void ArrayManager::receiveDataFromDevice() {
	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(host_C, dev_C, memorySizeForArray, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}

void ArrayManager::freeArrays() {
	free(host_A);
    free(host_B);
    free(host_C);
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
	host_A = NULL;
	host_B = NULL;
	host_C = NULL;
	dev_A = NULL;
	dev_B = NULL;
	dev_C = NULL;
}