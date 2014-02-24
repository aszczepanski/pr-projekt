#ifndef MULTIPLICATION_RUNNER_H_
#define MULTIPLICATION_RUNNER_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "array_manager.h"

class Multiplicator;

class MultiplicationRunner {
public:
	void runTest(const size_t arraySize, const size_t blockSize, Multiplicator* multiplicator);
protected:
	void synchronizeDevice();
	void testResult(const size_t arraySize);
	void resetDevice();

	cudaError_t cudaStatus;

private:
	void performTest(const size_t arraySize, const size_t blockSize, Multiplicator* multiplicator);
	void performCleanup();

	ArrayManager arrayManager;

	static const float bArrayValue;
};

#endif  // TESTER_H_