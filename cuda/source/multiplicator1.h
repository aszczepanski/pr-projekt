#ifndef MULTIPLICATOR1_H_
#define MULTIPLICATOR1_H_

#include "multiplicator.h"

class Multiplicator1 : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR1_H_