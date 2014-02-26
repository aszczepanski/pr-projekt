#include "multiplicator_factory.h"
#include "multiplicator1.h"
#include "multiplicator2.h"
#include "multiplicator3.h"
#include "multiplicator4a.h"
#include "multiplicator4b.h"
#include "multiplicator5.h"

#include <stdexcept>

Multiplicator* MultiplicatorFactory::getMultiplicator(const std::string& key) {
	if (multiplicators.find(key) != multiplicators.end()) {
		return multiplicators[key];
	} else {
		if (key == "mul1") {
			multiplicators[key] = new Multiplicator1();
			return multiplicators[key];
		} else if (key == "mul2") {
			multiplicators[key] = new Multiplicator2();
			return multiplicators[key];
		} else if (key == "mul3") {
			multiplicators[key] = new Multiplicator3();
			return multiplicators[key];
		} else if (key == "mul4a") {
			multiplicators[key] = new Multiplicator4a();
			return multiplicators[key];
		} else if (key == "mul4b") {
			multiplicators[key] = new Multiplicator4b();
			return multiplicators[key];
		} else if (key == "mul5") {
			multiplicators[key] = new Multiplicator5();
			return multiplicators[key];
		} else {
			throw std::invalid_argument("Unknown key.");
		}
	}
}

MultiplicatorFactory::~MultiplicatorFactory() {
	for (std::map<std::string, Multiplicator*>::iterator it = multiplicators.begin();
		 it != multiplicators.end(); it++) {
			 delete it->second;
			 it->second = NULL;
	}
}