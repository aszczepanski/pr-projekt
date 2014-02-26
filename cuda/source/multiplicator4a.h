#ifndef MULTIPLICATOR4A_H_
#define MULTIPLICATOR4A_H_

#include "multiplicator.h"

class Multiplicator4a : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR4A_H_