#ifndef MULTIPLICATOR5_H_
#define MULTIPLICATOR5_H_

#include "multiplicator.h"

class Multiplicator5 : public Multiplicator {
	void launchKernel(const size_t arraySize, const size_t blockSize, ArrayManager* arrayManager);
};

#endif  // MULTIPLICATOR5_H_