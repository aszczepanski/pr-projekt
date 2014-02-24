#include "multiplicator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Multiplicator::clearErrorFlag() {
	cudaStatus = cudaSuccess;
}