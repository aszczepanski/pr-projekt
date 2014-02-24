#include "applicaton.h"

#include <stdio.h>

#include "multiplication_runner.h"
#include "multiplicator_factory.h"

void Application::run() {
	printf("Application running\n");

	runTestsVariousBlockSize(multiplicatorFactory.getMultiplicator("mul1"));
	runTestsVariousBlockSize(multiplicatorFactory.getMultiplicator("mul2"));
	runTestsVariousBlockSize(multiplicatorFactory.getMultiplicator("mul3"));

	runTestsConstantBlockSize(multiplicatorFactory.getMultiplicator("mul3"));
	runTestsConstantBlockSize(multiplicatorFactory.getMultiplicator("mul4"));
	runTestsConstantBlockSize(multiplicatorFactory.getMultiplicator("mul5"));

}

void Application::runTestsVariousBlockSize(Multiplicator* multiplicator) {
	multiplicationRunner.runTest(176, 8, multiplicator);
	multiplicationRunner.runTest(352, 8, multiplicator);
	multiplicationRunner.runTest(528, 8, multiplicator);
	multiplicationRunner.runTest(176, 16, multiplicator);
	multiplicationRunner.runTest(352, 16, multiplicator);
	multiplicationRunner.runTest(528, 16, multiplicator);
	multiplicationRunner.runTest(176, 22, multiplicator);
	multiplicationRunner.runTest(352, 22, multiplicator);
	multiplicationRunner.runTest(528, 22, multiplicator);
}

void Application::runTestsConstantBlockSize(Multiplicator* multiplicator) {
	multiplicationRunner.runTest(64, 16, multiplicator);
	multiplicationRunner.runTest(128, 16, multiplicator);
	multiplicationRunner.runTest(256, 16, multiplicator);
	multiplicationRunner.runTest(384, 16, multiplicator);
	multiplicationRunner.runTest(512, 16, multiplicator);
}