#ifndef MULTIPLICATOR3_H_
#define MULTIPLICATOR3_H_

#include "multiplicator.h"

class Multiplicator3 : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR3_H_