#ifndef MULTIPLICATOR_H_
#define MULTIPLICATOR_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ArrayManager;

class Multiplicator {
public:
	virtual void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager) = 0;

protected:
	void clearErrorFlag();

	cudaError_t cudaStatus;
};

#endif  // MULTIPLICATOR_H_