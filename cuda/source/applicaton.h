#ifndef APPLICATION_H_
#define APPLICATION_H_

#include "multiplicator_factory.h"
#include "multiplication_runner.h"

class Multiplicator;

class Application {
public:
	void run();

private:
	void runTestsVariousBlockSize(Multiplicator* multiplicator);
	void runTestsConstantBlockSize(Multiplicator* multiplicator);

	MultiplicatorFactory multiplicatorFactory;
	MultiplicationRunner multiplicationRunner;
};

#endif  // APPLICATION_H_