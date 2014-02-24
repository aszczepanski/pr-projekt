#include "multiplication_runner.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_exceptions.h"
#include "multiplicator.h"

const float MultiplicationRunner::bArrayValue = 0.01f;

void MultiplicationRunner::runTest(const size_t arraySize, const size_t blockSize, Multiplicator* multiplicator) {
	performTest(arraySize, blockSize, multiplicator);
	performCleanup();	
}

void MultiplicationRunner::performTest(const size_t arraySize, const size_t blockSize, Multiplicator* multiplicator) {
	try {
		arrayManager.allocateAndInitializeArrays(arraySize);
		arrayManager.sendDataToDevice();
		multiplicator->launchKernel(arraySize, blockSize, &arrayManager);
		synchronizeDevice();
		arrayManager.receiveDataFromDevice();
		testResult(arraySize);
	} catch (CudaError& e) {
		fprintf(stderr, "CudaError: %s\n", e.what());
	}
}

void MultiplicationRunner::performCleanup() {
	try {
		arrayManager.freeArrays();
		synchronizeDevice();
		resetDevice();
	} catch (CudaError& e) {
		fprintf(stderr, "CudaError: %s\n", e.what());
	}
}

void MultiplicationRunner::synchronizeDevice() {   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}

void MultiplicationRunner::testResult(const size_t arraySize) {
	float* host_C = arrayManager.pointerToHost_C();

	printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula 
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps 
    double eps = 1.e-6 ; // machine zero
    for (int i = 0; i < (int)(arraySize * arraySize); i++) {
		double abs_err = fabs(host_C[i] - (i/arraySize+i%arraySize));
        double dot_length = arraySize;
        double abs_val = fabs(host_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, host_C[i], arraySize*bArrayValue, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
}

void MultiplicationRunner::resetDevice() {
	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        throw CudaError(cudaGetErrorString(cudaGetLastError()));
    }
}
