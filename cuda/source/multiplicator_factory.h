#ifndef MULTIPLICATOR_FACTORY_H_
#define MULTIPLICATOR_FACTORY_H_

#include "multiplicator.h"
#include <map>
#include <string>

class MultiplicatorFactory {
public:
	Multiplicator* getMultiplicator(const std::string& key);

	~MultiplicatorFactory();
private:
	std::map<std::string, Multiplicator*> multiplicators;
};

#endif  // MULTIPLICATOR_FACTORY_H_