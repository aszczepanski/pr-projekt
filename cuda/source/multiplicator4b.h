#ifndef MULTIPLICATOR4B_H_
#define MULTIPLICATOR4B_H_

#include "multiplicator.h"

class Multiplicator4b : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR4B_H_