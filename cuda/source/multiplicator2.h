#ifndef MULTIPLICATOR2_H_
#define MULTIPLICATOR2_H_

#include "multiplicator.h"

class Multiplicator2 : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR2_H_