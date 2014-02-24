#ifndef MULTIPLICATOR4_H_
#define MULTIPLICATOR4_H_

#include "multiplicator.h"

class Multiplicator4 : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR4_H_