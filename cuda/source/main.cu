#include "application.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int main() {
	Application application;
	application.run();

    return 0;
}