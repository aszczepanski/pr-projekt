#ifndef ARRAY_MANAGER_H_
#define ARRAY_MANAGER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ArrayManager {
public:
	void allocateAndInitializeArrays(const size_t arraySize);
	void freeArrays();

	void sendDataToDevice();
	void receiveDataFromDevice();

	size_t getMemorySizeForArray() { return memorySizeForArray;	}

	float* pointerToHost_A() { return host_A; }
	float* pointerToHost_B() { return host_B; }
	float* pointerToHost_C() { return host_C; }
	float* pointerToDev_A() { return dev_A; }
	float* pointerToDev_B() { return dev_B; }
	float* pointerToDev_C() { return dev_C; }

private:
	void initializeArrayWithConstValue(float* array, size_t size, const float value);
	void initializeArrayWithOnesOnDiagonal(float* array, size_t size);
	void initializeArrayWithSumIJ(float* array, size_t size);

	void allocateHostArrays();
	void allocateDeviceArrays();

	float* host_A;
	float* host_B;
	float* host_C;
	float* dev_A;
	float* dev_B;
	float* dev_C;

	size_t memorySizeForArray;

	cudaError_t cudaStatus;
};

#endif  // ARRAY_MANAGER_H_